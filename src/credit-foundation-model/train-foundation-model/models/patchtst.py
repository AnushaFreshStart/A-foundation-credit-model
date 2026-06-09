"""
patchtst.py — PatchTST-Only Credit Foundation Model
=====================================================
Pure PatchTST: Patch embedding + Transformer Encoder.
Best for capturing local temporal patterns.

Architecture:
    StepEmbedding(sum) → PatchEmbedding → TransformerEncoder → Head
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from components import (
    StepEmbedding, PatchEmbedding, AttentionPooling,
    MaskedPatchHead, JointHead, MultiTaskClassificationHead,
)
sys.path.insert(0, str(Path(__file__).parent.parent))


class PatchTSTModel(nn.Module):
    """PatchTST-only architecture — channel-independent patching + Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.embed_dim

        self.step_embed = StepEmbedding(
            config.vocab_size, D, config.step_width, mode="sum", dropout=config.dropout,
        )

        self.n_patches = config.max_seq_len // config.patch_size
        self.patch_embed = PatchEmbedding(D, config.patch_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=config.n_heads, dim_feedforward=config.ff_dim,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.attn_pool = AttentionPooling(D)
        self.patch_expand = nn.Linear(D, config.patch_size * D)

        self.pretrain_head = MaskedPatchHead(D, config.vocab_size)
        self.joint_head = JointHead(D)
        self.finetune_head = MultiTaskClassificationHead(D)

    def _encode(self, token_ids, attention_mask):
        embedded, _ = self.step_embed(token_ids)
        patches = self.patch_embed(embedded)
        B = attention_mask.size(0)
        pm = attention_mask[:, :self.n_patches * self.config.patch_size]
        pm = pm.reshape(B, self.n_patches, self.config.patch_size).any(dim=-1)
        encoded = self.transformer(patches, src_key_padding_mask=~pm.bool())
        return encoded, pm

    def forward(self, token_ids, attention_mask, stage="pretrain"):
        encoded, pm = self._encode(token_ids, attention_mask)
        if stage == "pretrain":
            exp = self.patch_expand(encoded)
            B, P, _ = exp.shape
            step_repr = exp.reshape(B, P * self.config.patch_size, self.config.embed_dim)
            return {"logits": self.pretrain_head(step_repr)}
        elif stage == "joint":
            exp = self.patch_expand(encoded)
            B, P, _ = exp.shape
            step_repr = exp.reshape(B, P * self.config.patch_size, self.config.embed_dim)
            return self.joint_head(step_repr)
        else:
            return self.finetune_head(self.attn_pool(encoded, pm))

    def get_embeddings(self, token_ids, attention_mask):
        encoded, pm = self._encode(token_ids, attention_mask)
        return self.attn_pool(encoded, pm)
