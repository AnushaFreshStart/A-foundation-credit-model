"""
lstm_baseline.py — LSTM Baseline Credit Foundation Model
=========================================================
Traditional deep learning baseline using bidirectional LSTM
with attention pooling. Non-transformer for comparison.

Architecture:
    StepEmbedding(sum) → BiLSTM → AttentionPooling → Head
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from components import (
    StepEmbedding, AttentionPooling,
    MaskedPatchHead, JointHead, MultiTaskClassificationHead,
)
sys.path.insert(0, str(Path(__file__).parent.parent))


class LSTMBaselineModel(nn.Module):
    """Bidirectional LSTM baseline — no transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.embed_dim

        self.step_embed = StepEmbedding(
            config.vocab_size, D, config.step_width, mode="sum", dropout=config.dropout,
        )

        self.lstm = nn.LSTM(
            input_size=D, hidden_size=D // 2, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=config.dropout,
        )

        self.attn_pool = AttentionPooling(D)

        self.pretrain_head = MaskedPatchHead(D, config.vocab_size)
        self.joint_head = JointHead(D)
        self.finetune_head = MultiTaskClassificationHead(D)

    def _encode(self, token_ids, attention_mask):
        embedded, _ = self.step_embed(token_ids)
        lstm_out, _ = self.lstm(embedded)  # (B, T, D)
        return lstm_out, attention_mask

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
