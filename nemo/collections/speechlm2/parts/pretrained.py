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
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule, SoundProjection
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


def load_pretrained_automodel(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, **kwargs):
    """
    Load a causal LM using NeMo Automodel (``NeMoAutoModelForCausalLM``).

    Automodel is a drop-in HuggingFace replacement that provides Liger kernel +
    SDPA attention optimizations and model-type-aware parallelization.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture
    with the checkpoint, but is randomly initialized.

    Extra ``kwargs`` (e.g. ``device_mesh``, ``distributed_config``, ``moe_mesh``,
    ``moe_config``) are forwarded to the underlying ``from_pretrained`` /
    ``from_config`` call so that parallelization happens during loading.
    """
    from nemo_automodel import NeMoAutoModelForCausalLM

    if pretrained_weights:
        return NeMoAutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype, trust_remote_code=True, **kwargs)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
        return NeMoAutoModelForCausalLM.from_config(config, torch_dtype=dtype, **kwargs)


def update_perception_output_dim(model):
    """
    Align the perception module's output projection with the actual LLM hidden size.

    When the LLM is loaded after the perception module (deferred init in
    ``configure_model``), the projection layer may have been created with an
    ``output_dim`` from the YAML config that doesn't match the LLM.  This
    helper replaces ``perception.proj`` with a correctly-sized ``nn.Linear``
    when the dimensions disagree.
    """
    hidden_size = model.llm.config.hidden_size
    proj = model.perception.proj
    if isinstance(proj, torch.nn.Linear) and proj.out_features != hidden_size:
        model.perception.proj = torch.nn.Linear(proj.in_features, hidden_size, bias=proj.bias is not None)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def _is_extracted_encoder_dir(path: str) -> bool:
    """Check whether ``path`` is an extracted encoder directory (safetensors + NeMo config)."""
    p = Path(path)
    if not p.is_dir() or not (p / "config.json").exists():
        return False
    with open(p / "config.json") as f:
        config = json.load(f)
    return "preprocessor" in config and "encoder" in config


def _load_safetensors_dir(directory: str | Path) -> Dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint directory into a single state dict.

    Reads ``model.safetensors.index.json`` to discover shard files, then loads
    and merges all shards.  Falls back to a single ``model.safetensors`` if no
    index file is present.
    """
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict = {}
        for shard_file in shard_files:
            state_dict.update(load_file(str(directory / shard_file), device="cpu"))
        return state_dict
    # Single-file fallback
    single = directory / "model.safetensors"
    if single.exists():
        return load_file(str(single), device="cpu")
    raise FileNotFoundError(f"No safetensors files found in {directory}")


def _setup_from_extracted(model: torch.nn.Module):
    """Set up ``AudioPerceptionModule`` from an extracted VL encoder directory.

    Loads the NeMo-compatible config, builds the perception module, loads
    encoder weights, and optionally replaces the projection with
    ``SoundProjection`` if ``model.cfg.pretrained_proj`` is set.
    """
    encoder_dir = Path(model.cfg.pretrained_asr)

    # 1. Load config and populate perception cfg
    with open(encoder_dir / "config.json") as f:
        config = json.load(f)

    with open_dict(model.cfg):
        model.cfg.perception.preprocessor = OmegaConf.create(config["preprocessor"])
        model.cfg.perception.encoder = OmegaConf.create(config["encoder"])
        if model.llm is not None:
            model.cfg.perception.output_dim = model.llm.config.hidden_size

    # 2. Build AudioPerceptionModule
    model.perception = AudioPerceptionModule(model.cfg.perception).train()

    # 3. Load encoder weights (strict=False: preprocessor buffers, modality_adapter, proj are missing — expected)
    encoder_state = _load_safetensors_dir(encoder_dir)
    result = model.perception.load_state_dict(encoder_state, strict=False)
    logging.info(
        f"Loaded extracted encoder: {len(encoder_state)} keys, "
        f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected"
    )

    # 4. Optionally load projection
    pretrained_proj = getattr(model.cfg, "pretrained_proj", None)
    if pretrained_proj and Path(pretrained_proj).is_dir():
        with open(Path(pretrained_proj) / "config.json") as f:
            proj_config = json.load(f)
        llm_hidden_size = model.llm.config.hidden_size if model.llm is not None else model.cfg.perception.output_dim
        model.perception.proj = SoundProjection(
            sound_hidden_size=proj_config["hidden_size"],
            projection_hidden_size=proj_config["projection_hidden_size"],
            llm_hidden_size=llm_hidden_size,
            bias=proj_config.get("projection_bias", False),
        )
        proj_state = _load_safetensors_dir(pretrained_proj)
        model.perception.proj.load_state_dict(proj_state)
        logging.info(f"Loaded extracted projection: {len(proj_state)} keys")


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel`` or an extracted VL encoder directory.
    The result is assigned to ``model.perception`` attribute and is trainable.
    """
    if pretrained_weights:
        pretrained_asr = model.cfg.pretrained_asr
        if _is_extracted_encoder_dir(pretrained_asr):
            _setup_from_extracted(model)
        else:
            asr = load_pretrained_nemo(ASRModel, pretrained_asr).eval()
            with open_dict(model.cfg):
                model.cfg.perception.preprocessor = asr.cfg.preprocessor
                model.cfg.perception.encoder = asr.cfg.encoder
                if model.llm is not None:
                    model.cfg.perception.output_dim = model.llm.config.hidden_size
            model.perception = AudioPerceptionModule(model.cfg.perception).train()
            model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        model.perception = AudioPerceptionModule(model.cfg.perception).train()


def set_model_dict_for_partial_init(
    pretrained_dict: Dict[str, torch.Tensor], model_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Partially initialize a model's state dictionary with a pretrained state dictionary.
    This function safely copies compatible layers from a pretrained model into a new model,
    ignoring layers with mismatched shapes or missing keys.

    Steps:
        1. Remove layers from the pretrained dictionary if their shape does not match the target model.
        2. Keep only keys that exist in the target model.
        3. Update the model dictionary with the filtered pretrained weights.

    Args:
        pretrained_dict (Dict[str, torch.Tensor]):
            The state dictionary of the pretrained model.
        model_dict (Dict[str, torch.Tensor]):
            The state dictionary of the target model to be partially initialized.

    Returns:
        Dict[str, torch.Tensor]:
            The updated model state dictionary with compatible layers loaded from the pretrained dictionary.

    Example:
        >>> model_dict = model.state_dict()
        >>> pretrained_dict = load_checkpoint("pretrained_model.ckpt")
        >>> model_dict = set_model_dict_for_partial_init(pretrained_dict, model_dict)
        >>> model.load_state_dict(model_dict)
    """
    # 1. Remove layers where pretrained shape differs from model shape
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(f" | > Layer with shape mismatch in the model definition: {k}")

    # 2. Keep only keys that exist in the target model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 3. Update model dictionary with filtered pretrained layers
    model_dict.update(pretrained_dict)
    logging.info(f" | > {len(pretrained_dict)} / {len(model_dict)} layers are restored.")

    return model_dict


def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from disk.

    Supports loading checkpoints stored in either PyTorch (`.ckpt`, `.pt`) or
    SafeTensors (`.safetensors`) formats. All parameters are loaded onto CPU
    regardless of the original device.

    Args:
        checkpoint_path (str):
            Path to the checkpoint file. If the filename contains `.safetensors`,
            it is loaded using the SafeTensors backend; otherwise, it is assumed
            to be a PyTorch checkpoint containing a `state_dict` field.

    Returns:
        dict:
            A state dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        checkpoint_state = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    return checkpoint_state
