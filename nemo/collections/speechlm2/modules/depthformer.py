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
Depthformer: RQ-transformer for generating Mimi audio codebook tokens from LLM hidden states.

Architecture based on liquid-audio's lfm2_audio.py depthformer implementation.
The depthformer takes LLM hidden states at audio output positions and autoregressively
predicts K codebook tokens per audio frame using teacher forcing during training.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import logging


# ---------------------------------------------------------------------------
# Lightweight transformer components (self-contained, no liquid-audio dependency)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ff_dim: int | None = None, out_init_scale: float = 1.0, multiple_of: int = 256):
        super().__init__()
        if ff_dim is None:
            ff_dim = int(2 * 4 * dim / 3)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

        std = 1.0 / math.sqrt(dim)
        nn.init.normal_(self.w1.weight, mean=0.0, std=std)
        nn.init.normal_(self.w3.weight, mean=0.0, std=std)
        std = out_init_scale / math.sqrt(ff_dim)
        nn.init.normal_(self.w2.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DepthformerAttention(nn.Module):
    """Multi-head attention with GQA support and rotary embeddings for the depthformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        gqa_dim: int = 8,
        out_init_scale: float = 1.0,
        norm_eps: float = 1e-5,
        max_seq_len: int = 64,
        theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gqa_dim = gqa_dim

        total_width = dim + 2 * self.head_dim * gqa_dim
        self.qkv_proj = nn.Linear(dim, total_width, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        std = 1.0 / math.sqrt(dim)
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=std)
        std = out_init_scale / math.sqrt(dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        # Precompute rotary embeddings
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2)[: self.head_dim // 2].float() / self.head_dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        kv_dim = self.head_dim * self.gqa_dim
        q, k, v = qkv.split([self.dim, kv_dim, kv_dim], dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.gqa_dim, self.head_dim)
        v = v.view(B, T, self.gqa_dim, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary embeddings
        if cache is not None:
            offset = cache[0].shape[1]
        else:
            offset = 0
        freqs = self.freqs_cis[offset : offset + T].to(q.device)
        q, k = _apply_rotary(q, k, freqs)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        new_cache = (k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and T > 1), enable_gqa=True)
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        out = self.out_proj(out)
        return out, new_cache


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings. q/k: (B, T, H, D), freqs_cis: (T, D/2) complex."""
    def rotate(x):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
        return torch.view_as_real(x_ * fc).flatten(-2).type_as(x)
    return rotate(q), rotate(k)


class DepthformerBlock(nn.Module):
    """Transformer block: MHA + RMSNorm + SwiGLU + residual connections."""

    def __init__(self, dim: int, out_init_scale: float = 1.0, norm_eps: float = 1e-5, **attn_kwargs):
        super().__init__()
        self.attn = DepthformerAttention(dim, out_init_scale=out_init_scale, norm_eps=norm_eps, **attn_kwargs)
        self.ffn = SwiGLU(dim, out_init_scale=out_init_scale)
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cache)
        h = h + x
        out = h + self.ffn(self.ffn_norm(h))
        return out, new_cache


class SharedEmbedding(nn.Module):
    """Embedding + RMSNorm + output projection, optionally weight-tied."""

    def __init__(self, dim: int, vocab_size: int, tie_embedding: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_norm = RMSNorm(dim, eps=norm_eps)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

        std = 1.0 / math.sqrt(dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        if tie_embedding:
            self.to_logits.weight = self.embedding.weight
        else:
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=std)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.to_logits(self.embedding_norm(hidden))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DepthformerConfig:
    """Configuration for the Depthformer RQ-transformer module."""
    num_codebooks: int = 8
    audio_vocab_size: int = 2049  # 2048 Mimi tokens + 1 EOAudio
    llm_hidden_size: int = 4096
    depthformer_dim: int = 1024
    depthformer_layers: int = 6
    depthformer_num_heads: int = 32
    depthformer_gqa_dim: int = 8
    tie_embeddings: bool = True
    codebook_weight_schedule: str = "log"  # "log" or "linear"
    semantic_codebook_factor: float = 100.0
    pretrained_depthformer: str | None = None  # path to extracted depthformer weights


# ---------------------------------------------------------------------------
# Depthformer
# ---------------------------------------------------------------------------


class Depthformer(nn.Module):
    """
    RQ-transformer that generates Mimi audio codebook tokens from LLM hidden states.

    During training, iterates over K codebooks with teacher forcing:
      for i in range(K):
        logits_i = depth_embeddings[i].get_logits(depthformer(df_in[:, i] + prev_emb))
        loss_i = cross_entropy(logits_i, targets[:, i])
        prev_emb = depth_embeddings[i].embed(targets[:, i])  # teacher forcing

    During inference, samples autoregressively per-frame through K codebooks.
    """

    def __init__(self, cfg: DepthformerConfig):
        super().__init__()
        self.cfg = cfg
        K = cfg.num_codebooks

        # Project LLM hidden states to per-codebook depthformer inputs
        self.depth_linear = nn.Linear(cfg.llm_hidden_size, cfg.depthformer_dim * K)

        # Transformer backbone
        scale = 1.0 / math.sqrt(2 * cfg.depthformer_layers)
        self.blocks = nn.ModuleList([
            DepthformerBlock(
                dim=cfg.depthformer_dim,
                out_init_scale=scale,
                num_heads=cfg.depthformer_num_heads,
                gqa_dim=cfg.depthformer_gqa_dim,
            )
            for _ in range(cfg.depthformer_layers)
        ])

        # Per-codebook shared embeddings (embedding + output head)
        self.depth_embeddings = nn.ModuleList([
            SharedEmbedding(
                dim=cfg.depthformer_dim,
                vocab_size=cfg.audio_vocab_size,
                tie_embedding=cfg.tie_embeddings,
            )
            for _ in range(K)
        ])

        # Audio output embedding: maps audio codes back into LLM hidden space
        # Each codebook gets its own slice of the embedding table, offset by codebook_offsets
        self.audio_embedding = SharedEmbedding(
            dim=cfg.llm_hidden_size,
            vocab_size=cfg.audio_vocab_size * K,
            tie_embedding=False,
        )

        # Codebook offsets for indexing into the fused audio_embedding
        self.register_buffer("codebook_offsets", torch.arange(K) * cfg.audio_vocab_size)

        # Per-codebook loss weights
        if cfg.codebook_weight_schedule == "log":
            weights = (torch.linspace(1, 0, K) * math.log(cfg.semantic_codebook_factor)).exp()
        else:
            weights = torch.ones(K)
            weights[0] *= cfg.semantic_codebook_factor
        self.register_buffer("audio_loss_weights", weights)

        # Load pretrained weights if provided
        if cfg.pretrained_depthformer:
            self._load_pretrained(cfg.pretrained_depthformer)

    def _load_pretrained(self, path: str):
        """Load pretrained depthformer weights from a safetensors directory."""
        p = Path(path)
        if not p.is_dir():
            logging.warning(f"Pretrained depthformer path {path} not found, using random init")
            return
        try:
            from safetensors.torch import load_file

            sf_path = p / "model.safetensors"
            if sf_path.exists():
                state_dict = load_file(str(sf_path), device="cpu")
            else:
                logging.warning(f"No model.safetensors found in {path}, using random init")
                return
            result = self.load_state_dict(state_dict, strict=False)
            logging.info(
                f"Loaded pretrained depthformer: {len(state_dict)} keys, "
                f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected"
            )
        except Exception as e:
            logging.warning(f"Failed to load pretrained depthformer from {path}: {e}")

    def _run_depthformer(self, x: torch.Tensor, cache: list | None = None) -> tuple[torch.Tensor, list]:
        """Run input through all transformer blocks. x: (B, T, D)."""
        if cache is None:
            cache = [None] * len(self.blocks)
        new_cache = []
        for block, block_cache in zip(self.blocks, cache):
            x, c = block(x, block_cache)
            new_cache.append(c)
        return x, new_cache

    def forward_train(
        self,
        llm_hidden_states: torch.Tensor,
        target_audio_codes: torch.Tensor,
        audio_output_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute audio loss during training with teacher forcing.

        Args:
            llm_hidden_states: (B, T, H) hidden states from the LLM
            target_audio_codes: (B, T, K) target Mimi codes at audio output positions
                (padded to full sequence length; only positions where audio_output_mask=True are valid)
            audio_output_mask: (B, T) boolean mask indicating audio output positions

        Returns:
            Scalar audio loss (weighted sum over codebooks)
        """
        K = self.cfg.num_codebooks

        # Extract hidden states at audio positions: flatten to (N, H)
        h = llm_hidden_states[audio_output_mask]  # (N, H)
        targets = target_audio_codes[audio_output_mask]  # (N, K)

        if h.shape[0] == 0:
            return h.new_tensor(0.0)

        # Project to per-codebook inputs: (N, K*D) -> (N, K, D)
        df_in = self.depth_linear(h).reshape(-1, K, self.cfg.depthformer_dim)

        # Sequential codebook prediction with teacher forcing
        prev_emb = torch.zeros_like(df_in[:, 0])  # (N, D)
        total_loss = h.new_tensor(0.0)

        for i in range(K):
            cur_input = (df_in[:, i] + prev_emb).unsqueeze(1)  # (N, 1, D)
            df_out, _ = self._run_depthformer(cur_input)  # (N, 1, D)
            logits = self.depth_embeddings[i].get_logits(df_out.squeeze(1))  # (N, V)
            loss_i = F.cross_entropy(logits, targets[:, i])
            total_loss = total_loss + self.audio_loss_weights[i] * loss_i
            # Teacher forcing: embed the ground-truth target
            prev_emb = self.depth_embeddings[i].embed(targets[:, i])  # (N, D)

        return total_loss

    @torch.no_grad()
    def forward_inference(
        self,
        llm_hidden_state: torch.Tensor,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample K codebook tokens for a single audio frame.

        Args:
            llm_hidden_state: (B, H) or (H,) hidden state from the LLM at an audio output position
            temperature: sampling temperature (None or <=0 for greedy)
            top_k: top-k filtering

        Returns:
            (B, K) or (K,) sampled codebook tokens
        """
        squeeze = llm_hidden_state.dim() == 1
        if squeeze:
            llm_hidden_state = llm_hidden_state.unsqueeze(0)

        K = self.cfg.num_codebooks
        B = llm_hidden_state.shape[0]

        df_in = self.depth_linear(llm_hidden_state).reshape(B, K, self.cfg.depthformer_dim)
        prev_emb = torch.zeros(B, self.cfg.depthformer_dim, device=llm_hidden_state.device, dtype=llm_hidden_state.dtype)
        cache = None

        greedy = temperature is None or temperature <= 0 or top_k == 1
        out_tokens = []

        for i in range(K):
            cur_input = (df_in[:, i] + prev_emb).unsqueeze(1)  # (B, 1, D)
            df_out, cache = self._run_depthformer(cur_input, cache)  # (B, 1, D)
            logits = self.depth_embeddings[i].get_logits(df_out.squeeze(1))  # (B, V)

            if greedy:
                tokens = logits.argmax(dim=-1)  # (B,)
            else:
                logits = logits / temperature
                if top_k is not None:
                    min_score = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < min_score, float("-inf"))
                probs = logits.softmax(dim=-1)
                tokens = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

            out_tokens.append(tokens)
            prev_emb = self.depth_embeddings[i].embed(tokens)  # (B, D)

        result = torch.stack(out_tokens, dim=-1)  # (B, K)
        return result.squeeze(0) if squeeze else result

    def embed_audio_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Embed audio codebook tokens into the LLM's hidden space.

        For teacher forcing during training, this creates embeddings of the *previous*
        timestep's audio codes that are fed back into the LLM input.

        Args:
            codes: (B, T, K) or (B, K) audio codebook tokens

        Returns:
            (B, T, H) or (B, H) embeddings, summed across codebooks
        """
        # Offset codes per codebook: each codebook indexes a different slice of audio_embedding
        offset_codes = codes + self.codebook_offsets  # broadcast: (..., K) + (K,)
        # Embed each codebook and sum: audio_embedding maps (audio_vocab_size * K) -> H
        embs = self.audio_embedding.embed(offset_codes)  # (..., K, H)
        return embs.sum(dim=-2)  # (..., H)
