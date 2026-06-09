"""
lightweight.py — Lightweight Transformer Credit Foundation Model
=================================================================
Minimal transformer: simple embedding + sinusoidal positional encoding
+ small Transformer Encoder. Fastest to train.

Architecture:
    StepEmbedding(sum) → PositionalEncoding → TransformerEncoder → Head
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from components import (
    StepEmbedding, SinusoidalPositionalEncoding, AttentionPooling,
    MaskedPatchHead, JointHead, MultiTaskClassificationHead,
)
sys.path.insert(0, str(Path(__file__).parent.parent))


class LightweightModel(nn.Module):
    """Minimal transformer — no patching, no VSN."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.embed_dim

        self.step_embed = StepEmbedding(
            config.vocab_size, D, config.step_width, mode="sum", dropout=config.dropout,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(D, config.max_seq_len + 10, config.dropout)

        # Use fewer layers for lightweight
        n_layers = max(1, min(config.n_layers, 2))
        n_heads = max(1, min(config.n_heads, 4))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=min(config.ff_dim, 128),
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_pool = AttentionPooling(D)

        self.pretrain_head = MaskedPatchHead(D, config.vocab_size)
        self.joint_head = JointHead(D)
        self.finetune_head = MultiTaskClassificationHead(D)

    def _encode(self, token_ids, attention_mask):
        embedded, _ = self.step_embed(token_ids)
        embedded = self.pos_encoding(embedded)
        encoded = self.transformer(embedded, src_key_padding_mask=~attention_mask.bool())
        return encoded, attention_mask

    def forward(self, token_ids, attention_mask, stage="pretrain"):
        encoded, mask = self._encode(token_ids, attention_mask)
        if stage == "pretrain":
            return {"logits": self.pretrain_head(encoded)}
        elif stage == "joint":
            return self.joint_head(encoded)
        else:
            return self.finetune_head(self.attn_pool(encoded, mask))

    def get_embeddings(self, token_ids, attention_mask):
        encoded, mask = self._encode(token_ids, attention_mask)
        return self.attn_pool(encoded, mask)
