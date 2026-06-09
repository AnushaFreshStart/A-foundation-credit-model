"""
models/__init__.py — Architecture Registry
============================================
Factory function to build any model architecture from config.
"""

from __future__ import annotations

import torch.nn as nn

from .patchtst import PatchTSTModel
from .tft import TFTModel
from .hybrid import HybridModel
from .lightweight import LightweightModel
from .lstm_baseline import LSTMBaselineModel


ARCHITECTURE_REGISTRY: dict[str, type[nn.Module]] = {
    "patchtst": PatchTSTModel,
    "tft": TFTModel,
    "hybrid": HybridModel,
    "lightweight": LightweightModel,
    "lstm_baseline": LSTMBaselineModel,
}


def build_model(config) -> nn.Module:
    """Build a model from config.architecture string."""
    arch = config.architecture
    if arch not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    cls = ARCHITECTURE_REGISTRY[arch]
    model = cls(config)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {arch} | {n_params:,} params ({n_train:,} trainable)")

    return model
