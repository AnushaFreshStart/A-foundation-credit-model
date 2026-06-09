"""
tft.py — Temporal Fusion Transformer (TFT) Credit Foundation Model
====================================================================
TFT-only: Variable Selection → GRN → LSTM → Multi-Head Attention → GRN.
Best for feature importance and interpretability.

Architecture:
    StepEmbedding(VSN) → GRN → BiLSTM → MultiHeadAttention → GRN → Head
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from components import (
    StepEmbedding, GatedResidualNetwork, AttentionPooling,
    MaskedPatchHead, JointHead, MultiTaskClassificationHead,
)
sys.path.insert(0, str(Path(__file__).parent.parent))


class TFTModel(nn.Module):
    """TFT-only architecture with LSTM encoder and interpretable attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.embed_dim

        self.step_embed = StepEmbedding(
            config.vocab_size, D, config.step_width, mode="vsn", dropout=config.dropout,
        )

        self.pre_lstm_grn = GatedResidualNetwork(D, D, D, config.dropout)

        self.lstm = nn.LSTM(
            input_size=D, hidden_size=D // 2, num_layers=min(config.n_layers, 2),
            batch_first=True, bidirectional=True, dropout=config.dropout if config.n_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=D, num_heads=config.n_heads, dropout=config.dropout, batch_first=True,
        )

        self.post_attn_grn = GatedResidualNetwork(D, D, D, config.dropout)
        self.attn_pool = AttentionPooling(D)

        self.pretrain_head = MaskedPatchHead(D, config.vocab_size)
        self.joint_head = JointHead(D)
        self.finetune_head = MultiTaskClassificationHead(D)

        self._last_vsn_weights: Optional[torch.Tensor] = None
        self._last_attn_weights: Optional[torch.Tensor] = None

    def _encode(self, token_ids, attention_mask):
        embedded, vsn_weights = self.step_embed(token_ids)
        self._last_vsn_weights = vsn_weights

        gated = self.pre_lstm_grn(embedded)
        lstm_out, _ = self.lstm(gated)

        key_pad_mask = ~attention_mask.bool()
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=key_pad_mask,
        )
        self._last_attn_weights = attn_weights

        encoded = self.post_attn_grn(attn_out + lstm_out)
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

    def get_variable_importance(self):
        return self._last_vsn_weights
