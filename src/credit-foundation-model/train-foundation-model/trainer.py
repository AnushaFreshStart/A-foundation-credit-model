"""
trainer.py — Training Orchestrator for Credit Foundation Model
================================================================
Runs 3-stage training pipeline (pretrain → joint → finetune) with
FP16 AMP, early stopping, checkpoint management, and embedding export.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from models import build_model
from dataset import CreditSequenceDataset, collate_fn
from losses import MaskedPredictionLoss, MultiObjectiveLoss, MultiTaskFineTuneLoss
from evaluator import ModelEvaluator


class CreditModelTrainer:
    """
    Unified training orchestrator.

    Supports all 5 architectures × 5 strategies with FP16 AMP,
    early stopping, and comprehensive evaluation.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {self.device}")

        self.model = build_model(config).to(self.device)
        self.evaluator = ModelEvaluator()
        self.history: dict[str, list] = {}

        # Resolve paths
        config.resolve_paths()

    def train(self) -> dict:
        """Run the configured training strategy."""
        stages = self._resolve_stages()
        results = {
            "config": self.config.to_dict(),
            "device": str(self.device),
            "stages": {},
            "training_curves": {},
        }

        total_t0 = time.perf_counter()

        for stage in stages:
            print(f"\n{'=' * 60}")
            print(f"  STAGE: {stage.upper()}")
            print(f"{'=' * 60}")

            stage_result = self._run_stage(stage)
            results["stages"][stage] = stage_result

        results["total_time_s"] = round(time.perf_counter() - total_t0, 2)

        # Export embeddings
        if self.config.save_embeddings and "finetune" in stages:
            print(f"\n{'=' * 60}")
            print(f"  EXPORTING EMBEDDINGS")
            print(f"{'=' * 60}")
            results["embeddings"] = self._export_embeddings()

        # Model parameter count
        results["total_params"] = sum(p.numel() for p in self.model.parameters())
        results["trainable_params"] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return results

    def _resolve_stages(self) -> list[str]:
        strategies = {
            "full": ["pretrain", "joint", "finetune"],
            "pretrain_only": ["pretrain"],
            "pretrain_finetune": ["pretrain", "finetune"],
            "finetune_only": ["finetune"],
            "joint_finetune": ["joint", "finetune"],
        }
        return strategies[self.config.strategy]

    def _run_stage(self, stage: str) -> dict:
        """Run a single training stage."""
        cfg = self.config

        # --- Build dataset ---
        mode_map = {"pretrain": "pretrain", "joint": "joint", "finetune": "finetune"}
        label_maps = None

        if stage == "finetune":
            label_maps = CreditSequenceDataset.load_label_maps(cfg.db_path)

        ds = CreditSequenceDataset(
            cfg.sequences_path, cfg.tokenizer_path,
            mode=mode_map[stage], label_maps=label_maps,
            patch_size=cfg.patch_size,
        )
        train_ds, val_ds = ds.oot_split(train_year_max=2024)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # --- Loss function ---
        if stage == "pretrain":
            loss_fn = MaskedPredictionLoss()
            n_epochs = cfg.pretrain_epochs
        elif stage == "joint":
            loss_fn = MultiObjectiveLoss(cfg.alpha, cfg.beta, cfg.gamma)
            n_epochs = cfg.joint_epochs
        else:
            loss_fn = MultiTaskFineTuneLoss(cfg.cure_weight, cfg.focal_gamma, cfg.focal_alpha)
            n_epochs = cfg.finetune_epochs

        # --- Optimizer & scheduler ---
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7,
        )

        # --- AMP ---
        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # --- Training loop ---
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        best_state = None
        train_losses = []
        val_losses = []

        t0 = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = self.model(ids, mask, stage=stage)
                    loss = self._compute_loss(loss_fn, outputs, batch, stage)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(round(avg_train_loss, 6))

            # --- Validate ---
            self.model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(self.device)
                    mask = batch["attention_mask"].to(self.device)

                    with torch.amp.autocast(
                        device_type=self.device.type, dtype=torch.float16, enabled=use_amp
                    ):
                        outputs = self.model(ids, mask, stage=stage)
                        loss = self._compute_loss(loss_fn, outputs, batch, stage)

                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            val_losses.append(round(avg_val_loss, 6))

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # Log progress
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss={avg_train_loss:.5f} | "
                f"val_loss={avg_val_loss:.5f} | "
                f"lr={lr:.2e} | "
                f"patience={patience_counter}/{patience} | "
                f"{elapsed:.1f}s"
            )

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best checkpoint
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        stage_time = time.perf_counter() - t0

        # --- Evaluate ---
        print(f"\n  Evaluating {stage}...")
        if stage == "pretrain":
            metrics = self.evaluator.evaluate_pretrain(self.model, val_loader, self.device)
        elif stage == "joint":
            metrics = self.evaluator.evaluate_joint(self.model, val_loader, self.device)
        else:
            metrics = self.evaluator.evaluate_finetune(self.model, val_loader, self.device)

        return {
            "metrics": metrics,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": round(best_val_loss, 6),
            "epochs_trained": len(train_losses),
            "stage_time_s": round(stage_time, 2),
        }

    def _compute_loss(self, loss_fn, outputs, batch, stage):
        """Compute loss based on stage and loss function type."""
        if stage == "pretrain":
            labels = batch["labels"].to(self.device)
            logits = outputs["logits"]
            return loss_fn(logits, labels)

        elif stage == "joint":
            labels = batch["labels"].to(self.device).float()
            mask = batch["attention_mask"].to(self.device)
            result = loss_fn(outputs, labels, mask)
            return result["loss"]

        else:  # finetune
            def_labels = batch["default_label"].to(self.device)
            cure_labels = batch["cure_label"].to(self.device)
            result = loss_fn(outputs, def_labels, cure_labels)
            return result["loss"]

    def _export_embeddings(self) -> dict:
        """Export loan-level embeddings + prediction probabilities."""
        cfg = self.config

        # Load full dataset in finetune mode for labels
        label_maps = CreditSequenceDataset.load_label_maps(cfg.db_path)
        ds = CreditSequenceDataset(
            cfg.sequences_path, cfg.tokenizer_path,
            mode="finetune", label_maps=label_maps,
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        self.model.eval()
        all_embeddings = []
        all_loan_ids = []
        all_def_probs = []
        all_cure_probs = []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)

                # Get embeddings
                emb = self.model.get_embeddings(ids, mask)  # (B, D)
                all_embeddings.append(emb.cpu().numpy())
                all_loan_ids.extend(batch["loan_ids"])

                # Get predictions
                outputs = self.model(ids, mask, stage="finetune")
                def_prob = torch.sigmoid(outputs["default_logit"].squeeze(-1)).cpu().numpy()
                cure_prob = torch.sigmoid(outputs["cure_logit"].squeeze(-1)).cpu().numpy()
                all_def_probs.append(def_prob)
                all_cure_probs.append(cure_prob)

        embeddings = np.concatenate(all_embeddings, axis=0)
        def_probs = np.concatenate(all_def_probs)
        cure_probs = np.concatenate(all_cure_probs)

        # Build DataFrame
        import pandas as pd

        embed_cols = {f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}
        df = pd.DataFrame({
            "loan_id": all_loan_ids,
            **embed_cols,
            "default_prob": np.round(def_probs, 6),
            "cure_prob": np.round(cure_probs, 6),
        })

        # Save
        output_path = Path(cfg.output_dir) / "embeddings.parquet"
        df.to_parquet(str(output_path), index=False)
        print(f"  OK Embeddings saved → {output_path.name} ({len(df):,} loans × {embeddings.shape[1]} dims)")

        return {
            "n_loans": len(df),
            "embed_dim": int(embeddings.shape[1]),
            "output_path": str(output_path),
            "default_prob_mean": round(float(def_probs.mean()), 4),
            "cure_prob_mean": round(float(cure_probs.mean()), 4),
        }

    def get_checkpoint(self) -> dict:
        """Return model checkpoint dict."""
        return {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "architecture": self.config.architecture,
        }
