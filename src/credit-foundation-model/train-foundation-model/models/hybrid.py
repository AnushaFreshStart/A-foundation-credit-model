"""
hybrid.py — Hybrid TFT + PatchTST Credit Foundation Model (Recommended)
=========================================================================
Combines TFT's Variable Selection Network for intelligent feature gating
with PatchTST's patching mechanism and Transformer Encoder for temporal
pattern capture.

Architecture:
    StepEmbedding(VSN) → GRN → PatchEmbedding → TransformerEncoder → Head
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from components import (
    StepEmbedding, PatchEmbedding, GatedResidualNetwork,
    AttentionPooling, MaskedPatchHead, JointHead,
    MultiTaskClassificationHead,
)

# Allow importing config
sys.path.insert(0, str(Path(__file__).parent.parent))


class HybridModel(nn.Module):
    """
    Hybrid TFT + PatchTST architecture.

    Combines:
    - TFT's Variable Selection Network for feature importance
    - GRN gating for non-linear feature interactions
    - PatchTST's patching + Transformer for temporal patterns
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.embed_dim

        # Step embedding with VSN
        self.step_embed = StepEmbedding(
            config.vocab_size, D, config.step_width,
            mode="vsn", dropout=config.dropout,
        )

        # GRN for post-VSN gating
        self.grn = GatedResidualNetwork(D, D, D, config.dropout)

        # Patch embedding
        self.n_patches = config.max_seq_len // config.patch_size
        self.patch_embed = PatchEmbedding(D, config.patch_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # Attention pooling for classification
        self.attn_pool = AttentionPooling(D)

        # Patch-to-step expansion for pre-training
        self.patch_expand = nn.Linear(D, config.patch_size * D)

        # Stage-specific output heads
        self.pretrain_head = MaskedPatchHead(D, config.vocab_size)
        self.joint_head = JointHead(D)
        self.finetune_head = MultiTaskClassificationHead(D)

        # Store last VSN weights for interpretability
        self._last_vsn_weights: Optional[torch.Tensor] = None

    def _encode(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder path: embed → VSN → GRN → patch → transformer."""
        # Step embedding with VSN: (B, T, D), (B, T, 5)
        embedded, vsn_weights = self.step_embed(token_ids)
        self._last_vsn_weights = vsn_weights

        # GRN gating
        gated = self.grn(embedded)  # (B, T, D)

        # Patch embedding: (B, n_patches, D)
        patches = self.patch_embed(gated)

        # Create patch-level attention mask
        B = attention_mask.size(0)
        patch_mask = attention_mask[:, :self.n_patches * self.config.patch_size]
        patch_mask = patch_mask.reshape(B, self.n_patches, self.config.patch_size)
        patch_mask = patch_mask.any(dim=-1)  # (B, n_patches) — patch valid if any step valid

        # Transformer encoding with mask
        src_key_padding_mask = ~patch_mask.bool()
        encoded = self.transformer(patches, src_key_padding_mask=src_key_padding_mask)

        return encoded, patch_mask

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stage: str = "pretrain",
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_ids: (B, max_seq_len, step_width)
            attention_mask: (B, max_seq_len)
            stage: 'pretrain' | 'joint' | 'finetune'

        Returns:
            dict of output tensors depending on stage
        """
        encoded, patch_mask = self._encode(token_ids, attention_mask)

        if stage == "pretrain":
            # Expand patches back to per-step: (B, n_patches, patch_size * D)
            expanded = self.patch_expand(encoded)
            B, P, _ = expanded.shape
            D = self.config.embed_dim
            step_repr = expanded.reshape(B, P * self.config.patch_size, D)
            logits = self.pretrain_head(step_repr)  # (B, T', vocab_size)
            return {"logits": logits}

        elif stage == "joint":
            expanded = self.patch_expand(encoded)
            B, P, _ = expanded.shape
            D = self.config.embed_dim
            step_repr = expanded.reshape(B, P * self.config.patch_size, D)
            return self.joint_head(step_repr)

        else:  # finetune
            pooled = self.attn_pool(encoded, patch_mask)  # (B, D)
            return self.finetune_head(pooled)

    def get_embeddings(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract loan-level embeddings: (B, embed_dim)."""
        encoded, patch_mask = self._encode(token_ids, attention_mask)
        return self.attn_pool(encoded, patch_mask)

    def get_variable_importance(self) -> Optional[torch.Tensor]:
        """Return last-computed VSN weights: (B, T, 5)."""
        return self._last_vsn_weights
