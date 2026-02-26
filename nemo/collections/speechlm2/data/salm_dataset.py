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
import logging
from itertools import groupby
from typing import Iterable, Union

import numpy as np
import torch
import torch.utils.data
from lhotse import CutSet, fastcopy
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    TextTurn,
    collate_conversation_audio_fault_tolerant,
)
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class SALMDataset(torch.utils.data.Dataset):
    """
    A dataset for Speech-Augmented Language Models (SALM) that processes multimodal conversations
    containing both text and audio turns.

    This dataset handles NeMoMultimodalConversation objects which combine text messages
    and audio segments in a conversational format. It uses audio_locator_tag in the text,
    where each such placeholder corresponds to an entire audio segment.

    Args:
        tokenizer (AutoTokenizer):
            Tokenizer for converting text to token IDs and vice versa. Must have a special
            audio_locator_tag token that will be replaced with audio embeddings during model's
            training step.
        audio_locator_tag (str, optional):
            The placeholder token string for input audio (e.g., ``"<|audio|>"``). Required when
            ``audio_out_locator_tag`` is set, to identify which placeholders belong to assistant
            audio turns.
        audio_out_locator_tag (str, optional):
            The placeholder token string for output audio (e.g., ``"<|audio_out|>"``). When set,
            enables S2S mode: assistant AudioTurns are separated from user audio and their
            placeholders are expanded to ``<|audio_start|> <|audio_out|> <|audio_end|>``.

    Returns:
        A dictionary with the following keys:
            - audios: Tensor of audio waveform samples [B_audio, T_samples]
            - audio_lens: Tensor of audio lengths [B_audio]
            - input_ids: Tensor of text token IDs [B, T_tokens], including audio_locator_tag tokens
            - loss_mask: Boolean tensor [B, T_tokens] indicating which tokens are part of the
                assistant's responses (True) and should be used for computing loss

        When ``audio_out_locator_tag`` is set, the dictionary may also contain:
            - target_audios: Tensor of target audio waveform samples [B_target, T_samples]
            - target_audio_lens: Tensor of target audio lengths [B_target]

    Notes:
        - Each audio_locator_tag token in input_ids corresponds to an audio segment in audios
        - The SALM model later replaces these audio_locator_tag tokens with encoded audio embeddings
        - The loss_mask identifies which tokens are part of the target sequences (assistant responses)
          and which are part of the source sequences (user prompts)
        - The input_ids and loss_mask will be expanded during model forward pass to account for
          the variable-length audio segments that replace each audio_locator_tag token
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        audio_locator_tag: str | None = None,
        audio_out_locator_tag: str | None = None,
        audio_start_tag: str | None = None,
        context_audio_locator_tag: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)
        self.audio_locator_tag = audio_locator_tag
        self.audio_out_locator_tag = audio_out_locator_tag
        self.audio_start_tag = audio_start_tag
        self.context_audio_locator_tag = context_audio_locator_tag

    def __getitem__(self, conversations: CutSet) -> dict | None:
        # Note: the function call below may filter out some or all conversations due to audio loading issues.
        # If all conversations are filtered out, we'll return None, and expect users to wrap this dataset
        # in ``nemo.collections.common.data.fallback.FallbackDataset`` to use the previous mini-batch instead.
        try:
            audios, audio_lens, conversations = collate_conversation_audio_fault_tolerant(conversations)
        except Exception as e:
            logging.warning(f"Error collating conversations: {e}")
            return None
        if not conversations:
            return None

        if self.audio_out_locator_tag is None:
            # Original mode: no audio output separation.
            return {
                "audios": audios,
                "audio_lens": audio_lens,
                "input_ids": left_collate_vectors(
                    [c.input_ids for c in conversations], padding_value=self.pad_id
                ),
                "loss_mask": left_collate_vectors(
                    [getattr(c, "mask", torch.empty(0)) for c in conversations], padding_value=0
                ).to(torch.bool),
                "conversations": drop_in_memory_data(conversations),
            }

        # S2S mode: separate context/user/assistant audio, expand audio placeholders.
        all_input_ids = []
        all_masks = []
        user_audio_global = []
        target_audio_global = []
        context_audio_global = []
        audio_offset = 0

        for conv in conversations:
            new_ids, new_mask, user_indices, assistant_indices, context_indices = (
                self._expand_assistant_audio(conv)
            )
            all_input_ids.append(new_ids)
            all_masks.append(new_mask)
            for idx in user_indices:
                user_audio_global.append(audio_offset + idx)
            for idx in assistant_indices:
                target_audio_global.append(audio_offset + idx)
            for idx in context_indices:
                context_audio_global.append(audio_offset + idx)
            audio_offset += sum(1 for t in conv.turns if isinstance(t, AudioTurn))

        result = {
            "input_ids": left_collate_vectors(all_input_ids, padding_value=self.pad_id),
            "loss_mask": left_collate_vectors(all_masks, padding_value=0).to(torch.bool),
            "conversations": drop_in_memory_data(conversations),
        }

        if user_audio_global:
            result["audios"] = audios[user_audio_global]
            result["audio_lens"] = audio_lens[user_audio_global]
        else:
            result["audios"] = audios.new_zeros(0, 0)
            result["audio_lens"] = audio_lens.new_zeros(0)

        if target_audio_global:
            result["target_audios"] = audios[target_audio_global]
            result["target_audio_lens"] = audio_lens[target_audio_global]

        if context_audio_global:
            result["context_audios"] = audios[context_audio_global]
            result["context_audio_lens"] = audio_lens[context_audio_global]

        return result

    def _expand_assistant_audio(
        self, conv: NeMoMultimodalConversation
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int]]:
        """Swap audio placeholders based on turn role.

        - Assistant AudioTurns: ``<|audio|>`` → ``<|audio_out|>`` (depthformer target)
        - System AudioTurns (when ``context_audio_locator_tag`` is set): ``<|audio|>`` → ``<|context_audio|>`` (Mimi continuous encoder)
        - User AudioTurns: keep ``<|audio|>`` (perception encoder)

        Returns:
            new_input_ids: 1-D int tensor with swapped tokens.
            new_mask: 1-D mask tensor (unchanged).
            user_audio_indices: indices (into the conversation's audio turn list) for user audio.
            assistant_audio_indices: indices for assistant audio.
            context_audio_indices: indices for context audio (system-role AudioTurns).
        """
        input_ids = torch.as_tensor(conv.input_ids).clone()
        mask = torch.as_tensor(getattr(conv, "mask", torch.zeros_like(input_ids)))

        audio_locator_id = self.tokenizer.token_to_id(self.audio_locator_tag)
        audio_out_id = self.tokenizer.token_to_id(self.audio_out_locator_tag)
        context_audio_id = (
            self.tokenizer.token_to_id(self.context_audio_locator_tag)
            if self.context_audio_locator_tag is not None
            else None
        )

        # Classify each AudioTurn by role.
        audio_turns = [t for t in conv.turns if isinstance(t, AudioTurn)]
        user_audio_indices = []
        assistant_audio_indices = []
        context_audio_indices = []
        for audio_idx, turn in enumerate(audio_turns):
            if turn.role == "assistant":
                assistant_audio_indices.append(audio_idx)
            elif turn.role == "system" and context_audio_id is not None:
                context_audio_indices.append(audio_idx)
            else:
                user_audio_indices.append(audio_idx)

        if not assistant_audio_indices and not context_audio_indices:
            return input_ids, mask, user_audio_indices, assistant_audio_indices, context_audio_indices

        # Find placeholder positions in input_ids (one per AudioTurn, in order).
        placeholder_positions = (input_ids == audio_locator_id).nonzero(as_tuple=True)[0].tolist()
        assert len(placeholder_positions) == len(audio_turns), (
            f"Mismatch: {len(placeholder_positions)} placeholders vs {len(audio_turns)} audio turns"
        )

        # Swap assistant placeholders to audio_out_locator_tag.
        for audio_idx in assistant_audio_indices:
            input_ids[placeholder_positions[audio_idx]] = audio_out_id

        # Swap system/context placeholders to context_audio_locator_tag.
        for audio_idx in context_audio_indices:
            input_ids[placeholder_positions[audio_idx]] = context_audio_id

        # Insert <|audio_start|> before each assistant audio placeholder.
        if self.audio_start_tag is not None and assistant_audio_indices:
            audio_start_id = self.tokenizer.token_to_id(self.audio_start_tag)
            # Recalculate placeholder positions after the swap (values changed but positions didn't).
            assistant_positions = sorted([placeholder_positions[idx] for idx in assistant_audio_indices])
            # Build new tensors with <|audio_start|> inserted before each assistant audio.
            new_ids_parts, new_mask_parts = [], []
            prev = 0
            for pos in assistant_positions:
                new_ids_parts.append(input_ids[prev:pos])
                new_mask_parts.append(mask[prev:pos])
                new_ids_parts.append(torch.tensor([audio_start_id], dtype=input_ids.dtype))
                new_mask_parts.append(mask[pos : pos + 1])  # same mask as audio placeholder (True)
                prev = pos
            new_ids_parts.append(input_ids[prev:])
            new_mask_parts.append(mask[prev:])
            input_ids = torch.cat(new_ids_parts)
            mask = torch.cat(new_mask_parts)

        return input_ids, mask, user_audio_indices, assistant_audio_indices, context_audio_indices


def left_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")


def drop_in_memory_data(conversations: CutSet) -> CutSet:
    def _drop(conversation: NeMoMultimodalConversation) -> NeMoMultimodalConversation:
        turns = []
        for t in conversation.turns:
            if isinstance(t, AudioTurn):
                t = fastcopy(t, cut=t.cut.drop_in_memory_data())
            turns.append(t)
        return fastcopy(conversation, turns=turns)

    return conversations.map(_drop, apply_fn=None)


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama2PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama2PromptFormatter
):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    if hasattr(example, "system_prompt"):
        turns[0]["role"] = "system_and_user"
        turns[0]["slots"]["system"] = example.system_prompt
    return prompt.encode_dialog(turns)
