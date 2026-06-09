"""
config.py — Training Configuration for Credit Foundation Model
================================================================
Dataclass-based configuration with pre-defined hyperparameter profiles,
JSON serialization, validation, and automatic path resolution.

Usage:
    cfg = TrainingConfig()                          # default profile
    cfg = TrainingConfig.load_profile('small')      # small profile
    cfg = TrainingConfig.from_dict({...})            # from JSON
    cfg.resolve_paths()                              # auto-resolve workspace paths
    cfg.validate()                                   # check compatibility
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Hyperparameter Profiles
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "embed_dim": 64, "n_heads": 4, "n_layers": 3,
        "patch_size": 4, "ff_dim": 256, "dropout": 0.1,
        "batch_size": 64, "learning_rate": 1e-4,
        "pretrain_epochs": 50, "joint_epochs": 30, "finetune_epochs": 20,
    },
    "small": {
        "embed_dim": 32, "n_heads": 2, "n_layers": 2,
        "patch_size": 4, "ff_dim": 128, "dropout": 0.1,
        "batch_size": 128, "learning_rate": 2e-4,
        "pretrain_epochs": 30, "joint_epochs": 20, "finetune_epochs": 15,
    },
    "large": {
        "embed_dim": 128, "n_heads": 8, "n_layers": 4,
        "patch_size": 6, "ff_dim": 512, "dropout": 0.1,
        "batch_size": 32, "learning_rate": 5e-5,
        "pretrain_epochs": 80, "joint_epochs": 50, "finetune_epochs": 30,
    },
    "fast": {
        "embed_dim": 32, "n_heads": 2, "n_layers": 1,
        "patch_size": 8, "ff_dim": 64, "dropout": 0.05,
        "batch_size": 128, "learning_rate": 3e-4,
        "pretrain_epochs": 10, "joint_epochs": 8, "finetune_epochs": 5,
    },
}

VALID_ARCHITECTURES = ("patchtst", "tft", "hybrid", "lightweight", "lstm_baseline")
VALID_STRATEGIES = ("full", "pretrain_only", "pretrain_finetune", "finetune_only", "joint_finetune")


@dataclass
class TrainingConfig:
    """Full training configuration for the Credit Foundation Model."""

    # -- Architecture & strategy --
    architecture: str = "hybrid"
    strategy: str = "full"
    profile: str = "default"

    # -- Model hyperparameters --
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 3
    patch_size: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    vocab_size: int = 53
    max_seq_len: int = 24
    step_width: int = 5

    # -- Training hyperparameters --
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    pretrain_epochs: int = 50
    joint_epochs: int = 30
    finetune_epochs: int = 20

    # -- Loss weights (Stage 2: multi-objective) --
    alpha: float = 1.0    # MSE
    beta: float = 0.5     # NLL
    gamma: float = 0.3    # Calibration

    # -- Loss weights (Stage 3: multi-task) --
    cure_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # -- Self-supervised --
    mask_prob: float = 0.15

    # -- Acceleration --
    use_amp: bool = True
    precision: str = "fp16"

    # -- Run management --
    save_embeddings: bool = True
    run_name: str = ""

    # -- Paths (auto-resolved via resolve_paths) --
    workspace_dir: str = ""
    db_path: str = ""
    sequences_path: str = ""
    tokenizer_path: str = ""
    output_dir: str = ""

    # ---------------------------------------------------------------
    # Class methods
    # ---------------------------------------------------------------

    @classmethod
    def load_profile(cls, name: str, **overrides) -> "TrainingConfig":
        """Create a config from a named profile with optional overrides."""
        if name not in PROFILES and name != "custom":
            raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
        params = copy.deepcopy(PROFILES.get(name, {}))
        params["profile"] = name
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Construct from a dictionary (e.g., deserialized JSON)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    # ---------------------------------------------------------------
    # Instance methods
    # ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """Load config from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def resolve_paths(self) -> None:
        """Auto-resolve paths relative to the workspace directory."""
        if not self.workspace_dir:
            self.workspace_dir = str(Path(__file__).parent.parent)

        ws = Path(self.workspace_dir)

        if not self.db_path:
            self.db_path = str(ws / "pipelines" / "credit_validate.db")
        if not self.sequences_path:
            self.sequences_path = str(ws / "tokenize-sequences-result" / "sequences.parquet")
        if not self.tokenizer_path:
            self.tokenizer_path = str(ws / "tokenize-sequences-result" / "tokenizer.json")
        if not self.output_dir:
            self.output_dir = str(Path(__file__).parent)

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of warnings (empty = OK)."""
        warnings = []

        if self.architecture not in VALID_ARCHITECTURES:
            raise ValueError(
                f"Invalid architecture '{self.architecture}'. "
                f"Choose from: {VALID_ARCHITECTURES}"
            )

        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Choose from: {VALID_STRATEGIES}"
            )

        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.max_seq_len % self.patch_size != 0:
            warnings.append(
                f"max_seq_len ({self.max_seq_len}) not divisible by patch_size ({self.patch_size}). "
                f"Last partial patch will be dropped."
            )

        n_patches = self.max_seq_len // self.patch_size
        if n_patches < 2:
            warnings.append(
                f"Only {n_patches} patch(es) with patch_size={self.patch_size}. "
                f"Consider smaller patch_size for better attention."
            )

        if self.learning_rate > 1e-2:
            warnings.append(f"Learning rate {self.learning_rate} is very high for transformers.")

        return warnings

    def summary(self) -> str:
        """Human-readable summary string."""
        params = sum(1 for _ in [])  # placeholder
        return (
            f"Config: arch={self.architecture}, strategy={self.strategy}, "
            f"profile={self.profile}\n"
            f"  Model: embed={self.embed_dim}, heads={self.n_heads}, "
            f"layers={self.n_layers}, patch={self.patch_size}\n"
            f"  Training: bs={self.batch_size}, lr={self.learning_rate}, "
            f"epochs={self.pretrain_epochs}/{self.joint_epochs}/{self.finetune_epochs}\n"
            f"  Loss: α={self.alpha}, β={self.beta}, γ={self.gamma}, "
            f"cure_w={self.cure_weight}\n"
            f"  AMP: {self.precision if self.use_amp else 'disabled'}"
        )
