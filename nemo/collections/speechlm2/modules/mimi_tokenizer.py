# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Frozen Mimi codec wrapper for on-the-fly audio tokenization during training.

Uses HuggingFace ``transformers.MimiModel`` (no liquid-audio dependency).
Mimi is lazy-loaded and cached on first call to avoid registering it as a submodule
(which would cause FSDP/optimizer interference). The wrapper handles resampling
to Mimi's expected 24kHz sample rate.
"""

import torch
import torch.nn.functional as F

from nemo.utils import logging

MIMI_SAMPLE_RATE = 24000


class MimiTokenizer:
    """
    Frozen wrapper around the HuggingFace Mimi audio codec for on-the-fly tokenization.

    Not an nn.Module — stored as a plain Python object to avoid FSDP/optimizer interference.
    All Mimi parameters are frozen and the model runs in eval mode.

    Usage:
        tokenizer = MimiTokenizer(mimi_model_id="kyutai/mimi")
        tokenizer.maybe_load(device=torch.device("cuda"))
        codes, code_lens = tokenizer.encode(audio, audio_lens, source_sample_rate=16000)
    """

    def __init__(self, mimi_model_id: str = "kyutai/mimi", num_codebooks: int = 8):
        self.mimi_model_id = mimi_model_id
        self.num_codebooks = num_codebooks
        self._model = None
        self._frame_rate: float | None = None

    def maybe_load(self, device: torch.device | str = "cuda"):
        """Lazy-load the Mimi model on first use."""
        if self._model is not None:
            return
        logging.info(f"Loading Mimi codec from {self.mimi_model_id}")
        from transformers import MimiModel

        self._model = MimiModel.from_pretrained(self.mimi_model_id).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        # Derive frame rate from config: sample_rate / frame_size
        # frame_size = product of all upsampling ratios * hop_length
        cfg = self._model.config
        self._frame_rate = cfg.sampling_rate / self.frame_size
        logging.info(
            f"Mimi loaded: sample_rate={cfg.sampling_rate}, "
            f"frame_rate={self._frame_rate}, "
            f"num_codebooks={cfg.num_quantizers}, "
            f"using first {self.num_codebooks} codebooks"
        )

    @property
    def model(self):
        assert self._model is not None, "Call maybe_load() before using MimiTokenizer"
        return self._model

    @property
    def sample_rate(self) -> int:
        return MIMI_SAMPLE_RATE

    @property
    def frame_rate(self) -> float:
        if self._frame_rate is not None:
            return self._frame_rate
        # Fallback before model is loaded
        return 12.5

    @property
    def frame_size(self) -> int:
        """Number of audio samples per codec frame."""
        if self._model is not None:
            cfg = self._model.config
            return int(cfg.sampling_rate / cfg.frame_rate)
        # Fallback: 24000 / 12.5 = 1920
        return 1920

    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        source_sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio waveforms into Mimi codebook tokens.

        Args:
            audio: (B, T_samples) raw audio waveform
            audio_lens: (B,) lengths in samples at source_sample_rate
            source_sample_rate: sample rate of the input audio

        Returns:
            codes: (B, K, T_frames) int64 codebook tokens (K = num_codebooks)
            code_lens: (B,) lengths in frames
        """
        self.maybe_load(audio.device)

        # Resample to Mimi's sample rate if needed
        if source_sample_rate != MIMI_SAMPLE_RATE:
            ratio = MIMI_SAMPLE_RATE / source_sample_rate
            audio = _resample(audio, source_sample_rate, MIMI_SAMPLE_RATE)
            audio_lens = (audio_lens.float() * ratio).long()

        # HF MimiModel.encode expects (B, C, T) input_values and (B, C, T) padding_mask
        audio_3d = audio.unsqueeze(1)  # (B, 1, T)

        # Build padding mask: 1 for valid samples, 0 for padding
        T = audio_3d.shape[-1]
        padding_mask = torch.arange(T, device=audio.device).unsqueeze(0) < audio_lens.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(1).float()  # (B, 1, T)

        # Encode
        encoder_outputs = self.model.encode(
            audio_3d, padding_mask, num_quantizers=self.num_codebooks
        )
        codes = encoder_outputs.audio_codes  # (B, K, T_frames)

        # Compute frame lengths
        fs = self.frame_size
        code_lens = torch.ceil(audio_lens.float() / fs).long()

        return codes, code_lens

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        code_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode Mimi codebook tokens back to audio waveforms.

        Args:
            codes: (B, K, T_frames) codebook tokens
            code_lens: (B,) optional frame lengths

        Returns:
            audio: (B, T_samples) reconstructed waveform at Mimi's sample rate (24kHz)
            audio_lens: (B,) lengths in samples
        """
        self.maybe_load(codes.device)

        decoder_outputs = self.model.decode(codes)
        audio = decoder_outputs.audio_values  # (B, T_samples)

        fs = self.frame_size
        if code_lens is not None:
            audio_lens = (code_lens * fs).long()
        else:
            audio_lens = torch.full((audio.shape[0],), audio.shape[1], device=audio.device, dtype=torch.long)

        return audio, audio_lens


def _resample(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    """Simple resampling using linear interpolation. audio: (B, T)."""
    if source_sr == target_sr:
        return audio
    ratio = target_sr / source_sr
    new_len = int(audio.shape[-1] * ratio)
    # Use interpolate on (B, 1, T) -> (B, 1, new_T)
    return F.interpolate(audio.unsqueeze(1), size=new_len, mode="linear", align_corners=False).squeeze(1)
