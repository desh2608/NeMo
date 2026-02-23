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
import warnings
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, Optional

import torch
from lhotse import CutSet
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_automodel,
    setup_speech_encoder,
    update_perception_output_dim,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType


class SALM(LightningModule, HFHubMixin):
    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.audio_out_locator_tag = self.cfg.get("audio_out_locator_tag", "<|audio_out|>")
        self.audio_loss_weight = self.cfg.get("audio_loss_weight", 1.0)

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        special_tokens = [self.audio_locator_tag]
        if self.cfg.get("depthformer") is not None:
            special_tokens.extend([self.audio_out_locator_tag, "<|audio_start|>", "<|audio_end|>"])
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.llm = None  # populated by configure_model
        self.perception = None  # populated by configure_model
        self.depthformer = None  # populated by configure_model (if depthformer config present)
        self._mimi = None  # lazy-loaded MimiTokenizer (not a submodule)

        self._use_fsdp = False
        self._use_tp = False

        if self.cfg.get("init_configure_model", False):
            self.configure_model()

    @property
    def embed_tokens(self):
        """Navigate to the LLM's embedding layer (kept inside the LLM)."""
        if self.llm is None:
            return None
        return self.llm.model.embed_tokens

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the LLM's embedding table.

        Uses ``F.embedding`` instead of calling the ``nn.Embedding`` module to
        avoid triggering FSDP2 pre-forward hooks (which lazily initialize the
        child before the root LLM module, causing a ``RuntimeError``).

        When the weight is a sharded ``DTensor`` (FSDP2), we ``full_tensor()``
        it first to all-gather the complete embedding table — the same operation
        FSDP2 performs inside the LLM's forward pass.
        """
        weight = self.embed_tokens.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        return torch.nn.functional.embedding(input_ids.to(weight.device), weight)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    @property
    def audio_out_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_out_locator_tag)

    @property
    def token_equivalent_duration(self) -> float:
        """
        Returns the audio duration corresponding to a single frame/token at the output of ``self.perception``.
        """
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    def _get_mimi(self):
        """Lazy-load and cache the Mimi tokenizer (not a submodule to avoid FSDP/optimizer interference)."""
        if self._mimi is None:
            from nemo.collections.speechlm2.modules.mimi_tokenizer import MimiTokenizer

            self._mimi = MimiTokenizer(
                mimi_model_id=self.cfg.mimi_model_id,
                num_codebooks=self.cfg.depthformer.num_codebooks,
            )
            self._mimi.maybe_load(device=self.device)
        return self._mimi

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        need_hidden_states = self.depthformer is not None and cache is None
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        if not isinstance(out, dict):
            # NeMo Automodel doesn't respect return_dict=True yet
            ans = {"logits": out}
        else:
            ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
            if cache is not None:
                ans["cache"] = out["past_key_values"]
            if need_hidden_states and "hidden_states" in out:
                ans["hidden_states"] = out["hidden_states"][-1]  # last layer
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens (via Mimi codec, if depthformer is active).
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Replace <|audio|> and <|audio_out|> placeholders with their respective embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """
        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        audio_embs, audio_emb_lens = self.perception(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        )
        audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]

        # Build the placeholder replacement dict
        placeholder_dict = {self.audio_locator_tag_id: audio_embs}

        # Prepare audio output embeddings if depthformer is active and target audio is present
        target_audio_codes_list = []
        has_audio_output = (
            self.depthformer is not None
            and "target_audios" in batch
            and batch["target_audios"] is not None
            and batch["target_audios"].shape[0] > 0
        )
        if has_audio_output:
            mimi = self._get_mimi()
            codes, code_lens = mimi.encode(
                batch["target_audios"], batch["target_audio_lens"],
                source_sample_rate=self.sampling_rate,
            )
            audio_out_embs = []
            for j in range(codes.shape[0]):
                T_frames = code_lens[j].item()
                codes_j = codes[j, :, :T_frames].T  # (T_frames, K)
                target_audio_codes_list.append(codes_j)
                # Shifted codes for teacher forcing: first frame gets zeros
                shifted = torch.zeros_like(codes_j)
                shifted[1:] = codes_j[:-1]
                audio_out_embs.append(self.depthformer.embed_audio_tokens(shifted))
            placeholder_dict[self.audio_out_locator_tag_id] = audio_out_embs

        # Zero out all placeholder IDs before embedding
        input_ids = batch["input_ids"]
        ids_to_embed = input_ids.clone()
        for pid in placeholder_dict:
            ids_to_embed = torch.where(ids_to_embed == pid, 0, ids_to_embed)
        text_embs = self._embed_tokens(ids_to_embed)

        # Single-pass replacement of all placeholder types
        target_ids_raw = input_ids.where(batch["loss_mask"], -100)
        input_embs, target_ids, attention_mask, placeholder_tags = replace_placeholders_and_build_targets(
            input_ids=input_ids,
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_replacement_dict=placeholder_dict,
            target_ids=target_ids_raw,
        )

        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]
        placeholder_tags = placeholder_tags[:, 1:]  # align with target

        if self._use_tp:
            tp_world_size = self.device_mesh["tp"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]
                placeholder_tags = placeholder_tags[:, :-remainder]

        result = {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

        # Build audio output mask and scatter target codes using placeholder_tags
        if has_audio_output:
            audio_output_mask = (placeholder_tags == self.audio_out_locator_tag_id)
            # Scatter target_audio_codes into a padded (B, T, K) tensor aligned with the sequence
            K = self.cfg.depthformer.num_codebooks
            target_audio_codes = torch.zeros(
                *placeholder_tags.shape, K, dtype=torch.long, device=placeholder_tags.device
            )
            all_codes = torch.cat(target_audio_codes_list, dim=0)  # (total_frames, K)
            target_audio_codes[audio_output_mask] = all_codes
            result["audio_output_mask"] = audio_output_mask
            result["target_audio_codes"] = target_audio_codes

        return result

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        if torch.distributed.get_rank() == 0:
            B, T = inputs["input_embeds"].shape[:2]
            print(f'bs={B} seqlen={T}')
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        # Audio loss via depthformer
        audio_loss = text_loss.new_tensor(0.0)
        if (
            self.depthformer is not None
            and "audio_output_mask" in inputs
            and "hidden_states" in forward_outputs
        ):
            audio_loss = self.depthformer.forward_train(
                forward_outputs["hidden_states"],
                inputs["target_audio_codes"],
                inputs["audio_output_mask"],
            )

        loss = text_loss + self.audio_loss_weight * audio_loss

        if torch.distributed.get_rank() == 0:
            print(f'text_loss={text_loss.detach().cpu().item():.4f} audio_loss={audio_loss.detach().cpu().item():.4f}')

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_val_audio_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        if self._partial_val_audio_losses:
            for name, vals in self._partial_val_audio_losses.items():
                val_audio_loss = torch.stack(vals).mean()
                self.log(f"val_audio_loss_{name}", val_audio_loss, on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_val_audio_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                text_loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            # Audio loss
            if (
                self.depthformer is not None
                and "audio_output_mask" in inputs
                and "hidden_states" in forward_outputs
            ):
                audio_loss = self.depthformer.forward_train(
                    forward_outputs["hidden_states"],
                    inputs["target_audio_codes"],
                    inputs["audio_output_mask"],
                )
                self._partial_val_audio_losses[name].append(audio_loss)

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(text_loss)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        """Override Lightning's gradient clipping to handle mixed FSDP device meshes.

        When automodel parallelizes the LLM, some parameters end up as DTensors
        on the ``(dp_replicate, dp_shard_cp)`` mesh while others may be on the
        flattened ``dp`` mesh.  PyTorch's ``clip_grad_norm_`` requires all norms
        to share the same mesh for ``torch.stack``.  We delegate to automodel's
        mesh-aware ``_clip_grad_norm_impl`` which groups parameters by
        ``(mesh_id, placements)`` and combines per-group norms as plain tensors.
        """
        if not self._use_fsdp or gradient_clip_val is None or gradient_clip_val <= 0:
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        from nemo_automodel.components.training.utils import _clip_grad_norm_impl

        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if params:
            _clip_grad_norm_impl(params, max_norm=gradient_clip_val)

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate LLM answers given text or mixed text+audio prompts.

        Example 1. High-level API using ``prompts`` to provide both text and audio::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [
            ...             {
            ...                 "role": "user",
            ...                 "content": f"Transcribe the following: {model.audio_locator_tag}",
            ...                 "audio": ["path/to/audio.wav"],
            ...             }
            ...         ]
            ...    ],
            ...    max_new_tokens=128,
            ... )

        You may also include a ``transformers.GenerationConfig`` object to customize decoding strategy::

            >>> answer_ids = model.generate(..., generation_config=GenerationConfig(do_sample=True, num_beams=5))

        Example 2. Lower-level API, using ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=[
            ...        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}],
            ...        [{"role": "user", "content": f"Transcribe the following in Polish: {model.audio_locator_tag}"}],
            ...    ],
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Example 3. Lower-level API, using pre-tokenized and pre-formatted ``prompts`` for the text part,
        and pre-loaded ``audio`` and ``audio_lens`` tensors::

            >>> answer_ids = model.generate(
            ...    prompts=prompts,  # torch.Tensor, int64, of shape (batch, num_tokens)
            ...    audios=audios,  # torch.Tensor, float32, of shape (batch, time)
            ...    audio_lens=audio_lens,  # torch.Tensor, int64, of shape (batch,)
            ...    max_new_tokens=128,
            ... )

        Inputs:
            prompts: batch of prompts Tensor or as list[dict] each in the following format
                [
                  # batch example id 0
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}]
                  # batch example id 1
                  [{"role": "user"}, "slots": {"message": f"Transcribe the following in Polish: {model.audio_locator_tag}"}]
                ]
                "role" is LLM-specific, you can pass multiple turns as well.
                If ``prompts`` is a Tensor, we assume it was already formatted in the relevant chat template
                and tokenized with the model's tokenizer.
            audios: Optional. Time-domain audio signal zero-padded batch of shape (B, T).
                The number of audios must correspond to the number of occurrences of <audio_locator_tag> in prompts.
                Each prompt can have multiple audios.
            audio_lens: Optional. Length of each audio example.
            generation_config: Optional HuggingFace GenerationConfig object.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.
        """
        # Encode prompt dicts into int token ids.
        if isinstance(prompts, torch.Tensor):
            tokens = prompts.to(self.device)
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(prompts, sampling_rate=self.sampling_rate, device=self.device)
            ) is not None:
                assert (
                    audios is None and audio_lens is None
                ), "Audios cannot be provided via ``prompts`` and ``audios``/``audio_lens`` arguments simultaneously."
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=prompt)["input_ids"] for prompt in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )
        if audios is not None:
            # Audio + text input for generation.
            # Prepare token embeddings and audio embeddings.
            tokens_to_embed = tokens.where(tokens != self.audio_locator_tag_id, 0)
            token_embeds = self._embed_tokens(tokens_to_embed)
            # TODO: temporary workaround to perform batch_size=1 inference for audio encoder
            #   due to accuracy issues at bs>1
            audio_embeds, audio_embed_lens = self.perception(audios, audio_lens)
            audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]
            # Insert audio embeddings into relevant positions in text embeddings.
            input_embeds, _, attention_mask, _ = replace_placeholders_and_build_targets(
                input_ids=tokens,
                embeds=token_embeds,
                padding_id=self.text_pad_id,
                placeholder_replacement_dict={self.audio_locator_tag_id: audio_embeds},
            )
            answer_tokens = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        else:
            # Text-only generation — embed_tokens stays in LLM, HF generate uses it natively.
            attention_mask = tokens != self.text_pad_id
            answer_tokens = self.llm.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(
        self,
        device_mesh=None,
        distributed_config=None,
        moe_config=None,
        moe_mesh=None,
    ) -> None:
        # Use provided device_mesh, or fall back to LightningModule property
        if device_mesh is not None:
            self._device_mesh = device_mesh
        else:
            device_mesh = self.device_mesh

        # Derive dtype from trainer precision (e.g. "bf16-true" -> bfloat16).
        dtype = torch.float32
        if self._trainer is not None:
            precision = str(self._trainer.precision)
            if "bf16" in precision:
                dtype = torch.bfloat16
            elif "16" in precision:
                dtype = torch.float16
        elif hasattr(self.cfg, 'torch_dtype') and self.cfg.torch_dtype is not None:
            td = self.cfg.torch_dtype
            dtype = getattr(torch, td) if isinstance(td, str) else td

        # Fall back to trainer.strategy for configs (Lightning training path)
        if distributed_config is None and self._trainer is not None:
            distributed_config = getattr(self._trainer.strategy, "distributed_config", None)
        if moe_mesh is None and self._trainer is not None:
            moe_mesh = getattr(self._trainer.strategy, "moe_mesh", None)
        if moe_config is None and self._trainer is not None:
            moe_config = getattr(self._trainer.strategy, "moe_config", None)

        automodel_kwargs = {}
        if device_mesh is not None:
            automodel_kwargs["device_mesh"] = device_mesh
            # automodel's instantiate_infrastructure unconditionally calls
            # .to_dict() on these configs, so we must always provide defaults.
            if distributed_config is None:
                from nemo_automodel.components.distributed.config import FSDP2Config

                distributed_config = FSDP2Config()
            if moe_config is None:
                from nemo_automodel.components.moe.config import MoEParallelizerConfig

                moe_config = MoEParallelizerConfig()
            automodel_kwargs["distributed_config"] = distributed_config
            automodel_kwargs["moe_config"] = moe_config
        if moe_mesh is not None:
            automodel_kwargs["moe_mesh"] = moe_mesh

        self.llm = load_pretrained_automodel(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
            dtype=dtype,
            **automodel_kwargs,
        )

        # Create perception module (must happen after LLM so output_dim matches)
        setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)

        # Fix projection dim for pretrained_weights=False (config output_dim may not match LLM)
        update_perception_output_dim(self)

        # Conditionally instantiate Depthformer for audio output
        if self.cfg.get("depthformer") is not None:
            from nemo.collections.speechlm2.modules.depthformer import Depthformer, DepthformerConfig

            df_cfg = DepthformerConfig(
                llm_hidden_size=self.llm.config.hidden_size,
                **{k: v for k, v in self.cfg.depthformer.items() if k != "llm_hidden_size"},
            )
            self.depthformer = Depthformer(df_cfg).to(dtype=dtype)

        # Note: LoRA is deferred — see salm-lora-future-work.md in memory

        if device_mesh is None:
            return

        if device_mesh["tp"].size() > 1:
            self._use_tp = True

        # Use the same FSDP mesh as automodel uses for the LLM so that
        # gradient clipping can torch.stack norms from all parameters.
        dim_names = device_mesh.mesh_dim_names
        if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
        elif "dp_shard_cp" in dim_names:
            fsdp_mesh = device_mesh["dp_shard_cp"]
        else:
            fsdp_mesh = device_mesh["dp"]

        if fsdp_mesh.size() > 1:
            self._use_fsdp = True
            self.perception = fully_shard(self.perception, mesh=fsdp_mesh)
            if self.depthformer is not None:
                self.depthformer = fully_shard(self.depthformer, mesh=fsdp_mesh)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }


def replace_placeholders_and_build_targets(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    padding_id: int,
    placeholder_replacement_dict: dict[int, list[torch.Tensor]],
    target_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Replaces each occurrence of placeholder token IDs in input_ids with the corresponding
    tensors from the replacements in the embeds tensor, and creates corresponding adjusted target_ids.

    Note: when padding is necessary, we apply left-padding to the examples not to introduce
        anomalies at generation time.

    Args:
      input_ids (Tensor): shape (batch, sequence_length); input token ids.
      embeds (Tensor): shape (batch, sequence_length, hidden_dim); embeddings for each token.
      padding_id (int): these IDs will be marked as ignore_index in target_ids.
      placeholder_replacement_dict (dict): mapping from placeholder_id -> list of replacement
          tensors. Each replacement tensor has shape (L_i, hidden_dim). Replacements are consumed
          in order of occurrence within the sequence, independently per placeholder type.
      target_ids (Tensor): shape (batch, sequence_length); target token ids.

    Returns:
      Tuple of four tensors:
        - Tensor of shape (batch, max_new_sequence_length, hidden_dim) corresponding to
          ``embeds`` after replacements.
        - Tensor of shape (batch, max_new_sequence_length) with adjusted target IDs where:
          * Original target values are preserved where input was not a placeholder or padding
          * Positions that were placeholders, padding, or added by replacements are set to -100
          Will be None if target_ids input was None.
        - Tensor of shape (batch, max_new_sequence_length) with attention padding masks
          updated to account for shape changes due to replacements.
        - Tensor of shape (batch, max_new_sequence_length) with placeholder tags: for each
          output position, contains the placeholder_id that produced it, or 0 for non-placeholder
          positions. Useful for building masks per placeholder type downstream.
    """
    placeholder_ids_set = set(placeholder_replacement_dict.keys())
    # Per-placeholder consumption index (global across batch, consumed in sequence order)
    consumption_idx = {pid: 0 for pid in placeholder_ids_set}

    batch_size, seq_len = input_ids.size()
    if target_ids is not None:
        assert target_ids.size() == input_ids.size(), "target_ids must have the same shape as input_ids"

    hidden_dim = embeds.size(2)
    device, dtype = embeds.device, embeds.dtype
    ignore_index = -100  # Standard ignore_index value for CrossEntropyLoss

    # Un-pad the tensors because we'll need to re-apply new padding after replacements anyway.
    input_ids, embeds, target_ids = _unpad_inputs(input_ids, embeds, target_ids, padding_id)

    output_sequences = []
    output_target_ids = []
    output_att_masks = []
    output_placeholder_tags = []

    for i in range(batch_size):
        # Find all placeholder positions (for any placeholder type) at once
        is_placeholder = torch.zeros(input_ids[i].shape[0], dtype=torch.bool, device=device)
        for pid in placeholder_ids_set:
            is_placeholder |= (input_ids[i] == pid)
        placeholder_positions = is_placeholder.nonzero(as_tuple=True)[0]

        # Handle the case with no placeholders more efficiently
        if len(placeholder_positions) == 0:
            output_sequences.append(embeds[i])
            if target_ids is not None:
                new_target_ids = target_ids[i].clone()
                new_target_ids[input_ids[i] == padding_id] = ignore_index
                output_target_ids.append(new_target_ids)
            output_att_masks.append(input_ids[i] != padding_id)
            output_placeholder_tags.append(torch.zeros(input_ids[i].shape[0], dtype=torch.long, device=device))
            continue

        # Build segments between placeholders
        segments = []  # For embeddings
        target_segments = []  # For target IDs
        att_masks = []
        tag_segments = []  # For placeholder tags
        prev_pos = 0

        for pos in placeholder_positions:
            # Add segment before placeholder (if any)
            if pos > prev_pos:
                segments.append(embeds[i][prev_pos:pos])
                if target_ids is not None:
                    segment_target_ids = target_ids[i][prev_pos:pos].clone()
                    segment_target_ids[segment_target_ids == padding_id] = ignore_index
                    target_segments.append(segment_target_ids)
                att_masks.append(input_ids[i][prev_pos:pos] != padding_id)
                tag_segments.append(torch.zeros(pos - prev_pos, dtype=torch.long, device=device))

            # Determine which placeholder type this is and get the replacement
            pid = input_ids[i][pos].item()
            rep = placeholder_replacement_dict[pid][consumption_idx[pid]]
            consumption_idx[pid] += 1

            segments.append(rep)
            target_segments.append(torch.full((rep.size(0),), ignore_index, dtype=torch.long, device=device))
            att_masks.append(torch.ones((rep.size(0),), dtype=torch.bool, device=device))
            tag_segments.append(torch.full((rep.size(0),), pid, dtype=torch.long, device=device))

            prev_pos = pos + 1  # Skip placeholder

        # Add remaining segment after last placeholder (if any)
        remaining_len = input_ids[i].shape[0]
        if prev_pos < remaining_len:
            segments.append(embeds[i][prev_pos:remaining_len])
            if target_ids is not None:
                segment_target_ids = target_ids[i][prev_pos:remaining_len].clone()
                segment_target_ids[segment_target_ids == padding_id] = ignore_index
                target_segments.append(segment_target_ids)
            att_masks.append(input_ids[i][prev_pos:remaining_len] != padding_id)
            tag_segments.append(torch.zeros(remaining_len - prev_pos, dtype=torch.long, device=device))

        # Concatenate all segments for this example
        output_sequences.append(torch.cat(segments, dim=0))
        output_att_masks.append(torch.cat(att_masks, dim=0))
        output_placeholder_tags.append(torch.cat(tag_segments, dim=0))
        if target_ids is not None:
            output_target_ids.append(torch.cat(target_segments, dim=0))

    # Verify all replacements were used
    for pid, idx in consumption_idx.items():
        expected = len(placeholder_replacement_dict[pid])
        if idx != expected:
            raise ValueError(
                f"Placeholder {pid}: expected {expected} replacements but used {idx}"
            )

    # Create padded output tensors
    max_seq_length = max(seq.size(0) for seq in output_sequences)
    output = torch.zeros(batch_size, max_seq_length, hidden_dim, device=device, dtype=dtype)
    if target_ids is not None:
        new_target_ids = torch.full((batch_size, max_seq_length), ignore_index, dtype=torch.long, device=device)
    else:
        new_target_ids = None
    attention_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=device)
    placeholder_tags = torch.zeros((batch_size, max_seq_length), dtype=torch.long, device=device)

    if target_ids is None:
        output_target_ids = repeat(None)
    for i, (seq, tgt, att, tags) in enumerate(
        zip(output_sequences, output_target_ids, output_att_masks, output_placeholder_tags)
    ):
        seq_len = seq.size(0)
        output[i, -seq_len:] = seq
        if tgt is not None:
            new_target_ids[i, -seq_len:] = tgt
        attention_masks[i, -seq_len:] = att
        placeholder_tags[i, -seq_len:] = tags

    return output, new_target_ids, attention_masks, placeholder_tags


def _unpad_inputs(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    padding_id: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    def first_index_not_value(tensor, value):
        mask = tensor != value
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() > 0:
            return indices[0].item()
        else:
            return -1

    input_ids_unpad, embeds_unpad = [], []
    target_ids_unpad = [] if target_ids is not None else None
    for i in range(input_ids.shape[0]):
        idx = first_index_not_value(input_ids[i], padding_id)
        input_ids_unpad.append(input_ids[i, idx:])
        embeds_unpad.append(embeds[i, idx:])
        if target_ids is not None:
            target_ids_unpad.append(target_ids[i, idx:])
    return input_ids_unpad, embeds_unpad, target_ids_unpad


def _resolve_audios_in_prompt(
    prompts: list[list[dict]], sampling_rate: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    from lhotse import Recording

    paths = []
    for conversation in prompts:
        for turn in conversation:
            if "audio" in turn:
                turn_audio = turn["audio"]
                if isinstance(turn_audio, (str, Path)):
                    turn_audio = [turn_audio]
                for p in turn_audio:
                    assert isinstance(p, (str, Path)), f"Invalid value under prompt key 'audio': {p}"
                    paths.append(p)
    if not paths:
        return None
    cuts = CutSet([Recording.from_file(p).to_cut() for p in paths])
    with torch.device("cpu"):  # workaround for a Lhotse issue when default device is CUDA during collation
        audio, audio_lens = cuts.resample(sampling_rate).load_audio(collate=True)
    return (
        torch.as_tensor(audio).to(device, non_blocking=True),
        torch.as_tensor(audio_lens).to(device, non_blocking=True),
    )
