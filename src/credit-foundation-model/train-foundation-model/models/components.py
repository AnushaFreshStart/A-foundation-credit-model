"""
components.py — Shared Neural Network Building Blocks
======================================================
Reusable PyTorch modules used across all model architectures:
  - GatedResidualNetwork (GRN) — TFT building block
  - VariableSelectionNetwork (VSN) — feature gating
  - StepEmbedding — token ID → dense vectors
  - PatchEmbedding — sequence → patch representations
  - AttentionPooling — sequence → single vector
  - Output Heads — stage-specific prediction layers
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gated Residual Network (GRN) — TFT building block
# ---------------------------------------------------------------------------
class GatedResidualNetwork(nn.Module):
    """
    GRN: Linear → ELU → Linear → Dropout → GLU gate + residual + LayerNorm.

    Args:
        input_dim:  Input feature dimension
        hidden_dim: Internal hidden dimension
        output_dim: Output dimension (defaults to input_dim)
        dropout:    Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Residual projection if dimensions differ
        self.residual_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_proj is None else self.residual_proj(x)

        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)

        # GLU gate: split into value and gate, apply sigmoid to gate
        h = F.glu(h, dim=-1)

        return self.layer_norm(h + residual)


# ---------------------------------------------------------------------------
# Variable Selection Network (VSN) — TFT-style feature gating
# ---------------------------------------------------------------------------
class VariableSelectionNetwork(nn.Module):
    """
    Processes N input features through individual GRNs, computes softmax
    attention weights, and returns a weighted combination.

    Args:
        n_variables: Number of input variables (5 for STEP_WIDTH)
        embed_dim:   Embedding dimension per variable
        dropout:     Dropout probability
    """

    def __init__(self, n_variables: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_variables = n_variables
        self.embed_dim = embed_dim

        # Individual GRN per variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(embed_dim, embed_dim, embed_dim, dropout)
            for _ in range(n_variables)
        ])

        # Weight computation GRN (operates on concatenated inputs)
        self.weight_grn = GatedResidualNetwork(
            n_variables * embed_dim, embed_dim, n_variables, dropout
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_variables, embed_dim)

        Returns:
            output:  (batch, seq_len, embed_dim) — weighted combination
            weights: (batch, seq_len, n_variables) — selection weights
        """
        B, T, N, D = x.shape

        # Process each variable through its GRN
        processed = []
        for i in range(N):
            processed.append(self.variable_grns[i](x[:, :, i, :]))  # (B, T, D)
        processed = torch.stack(processed, dim=2)  # (B, T, N, D)

        # Compute selection weights
        flat = x.reshape(B, T, N * D)  # (B, T, N*D)
        weights = self.weight_grn(flat)  # (B, T, N)
        weights = F.softmax(weights, dim=-1)  # (B, T, N)

        # Weighted combination
        output = (processed * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

        return output, weights


# ---------------------------------------------------------------------------
# Step Embedding — token IDs → dense vectors
# ---------------------------------------------------------------------------
class StepEmbedding(nn.Module):
    """
    Embeds each of the 5 token positions per time step and aggregates them.

    Args:
        vocab_size: Total vocabulary size (53)
        embed_dim:  Embedding dimension (64)
        step_width: Tokens per step (5)
        mode:       'sum' (simple addition) or 'vsn' (Variable Selection Network)
        dropout:    Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        step_width: int = 5,
        mode: str = "sum",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.step_width = step_width

        # Shared embedding table
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if mode == "vsn":
            self.vsn = VariableSelectionNetwork(step_width, embed_dim, dropout)
        else:
            self.vsn = None

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            token_ids: (batch, seq_len, step_width) integer token IDs

        Returns:
            output:  (batch, seq_len, embed_dim)
            weights: (batch, seq_len, step_width) or None if mode='sum'
        """
        # Embed all positions: (B, T, 5, D)
        embedded = self.embedding(token_ids)

        if self.mode == "vsn" and self.vsn is not None:
            return self.vsn(embedded)  # (B, T, D), (B, T, 5)
        else:
            return embedded.sum(dim=2), None  # (B, T, D), None


# ---------------------------------------------------------------------------
# Patch Embedding — sequence → patch representations
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    Segments a sequence into non-overlapping patches and projects them.

    For seq_len=24, patch_size=4 → 6 patches.

    Args:
        embed_dim:  Input/output embedding dimension
        patch_size: Number of time steps per patch
        max_patches: Maximum number of patches (for positional encoding)
    """

    def __init__(self, embed_dim: int, patch_size: int, max_patches: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * embed_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            patches: (batch, n_patches, embed_dim)
        """
        B, T, D = x.shape
        n_patches = T // self.patch_size

        # Truncate to exact multiple of patch_size
        x = x[:, :n_patches * self.patch_size, :]

        # Reshape into patches: (B, n_patches, patch_size * D)
        x = x.reshape(B, n_patches, self.patch_size * D)

        # Project and add positional encoding
        x = self.projection(x)
        x = x + self.pos_encoding[:, :n_patches, :]
        x = self.layer_norm(x)

        return x


# ---------------------------------------------------------------------------
# Attention Pooling — sequence → single vector
# ---------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling over the sequence dimension.

    Args:
        embed_dim: Feature dimension
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.scale = math.sqrt(embed_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, embed_dim)
            mask: (batch, seq_len) — 1=valid, 0=pad

        Returns:
            pooled: (batch, embed_dim)
        """
        # Attention scores: (B, 1, T)
        scores = torch.matmul(self.query.expand(x.size(0), -1, -1), x.transpose(1, 2))
        scores = scores / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, 1, T)
        pooled = torch.matmul(weights, x).squeeze(1)  # (B, D)

        return pooled


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (for lightweight model)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Output Heads
# ---------------------------------------------------------------------------
class MaskedPatchHead(nn.Module):
    """Stage 1: Predict masked event tokens via cross-entropy over vocab."""

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.proj(x)


class JointHead(nn.Module):
    """Stage 2: Multi-objective output (point prediction, mu, log_sigma)."""

    def __init__(self, embed_dim: int, output_dim: int = 1):
        super().__init__()
        self.mu_head = nn.Linear(embed_dim, output_dim)
        self.log_sigma_head = nn.Linear(embed_dim, output_dim)
        self.point_head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            dict with 'mu', 'log_sigma', 'point_pred' — each (batch, seq_len, 1)
        """
        return {
            "mu": self.mu_head(x),
            "log_sigma": self.log_sigma_head(x),
            "point_pred": self.point_head(x),
        }


class MultiTaskClassificationHead(nn.Module):
    """Stage 3: Dual binary classification (default + cure)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.default_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )
        self.cure_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, embed_dim) — pooled sequence representation
        Returns:
            dict with 'default_logit' and 'cure_logit' — each (batch, 1)
        """
        return {
            "default_logit": self.default_head(x),
            "cure_logit": self.cure_head(x),
        }
