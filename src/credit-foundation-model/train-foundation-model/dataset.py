"""
dataset.py — Universal Dataset for Credit Foundation Model Training
=====================================================================
Supports all training modes: pretrain (masked patches), joint (next-step),
finetune (multi-task classification with default + cure labels).

Usage:
    ds = CreditSequenceDataset(seq_path, tok_path, mode='finetune', label_maps=maps)
    train_ds, test_ds = ds.oot_split(train_year_max=2024)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import tokenizer constants from the tokenize-sequences-app
_TOK_APP = Path(__file__).parent.parent / "tokenize-sequences-app"
sys.path.insert(0, str(_TOK_APP))
from tokenizer import LoanTokenizer, MAX_SEQ_LEN, STEP_WIDTH


MASK_PROB = 0.15
EVENT_POS = 0  # position of event token within each step


class CreditSequenceDataset(Dataset):
    """
    Universal dataset for all training stages.

    Modes:
        pretrain:  Masked patch prediction (mask 15% of patches)
        joint:     Next-step prediction with continuous targets
        finetune:  Multi-task classification (default + cure labels)
    """

    def __init__(
        self,
        sequences_path: str | Path,
        tokenizer_path: str | Path,
        mode: Literal["pretrain", "joint", "finetune"] = "pretrain",
        label_maps: dict[str, dict[str, int]] | None = None,
        patch_size: int = 4,
        seed: int = 42,
    ):
        import pyarrow.parquet as pq

        self.path = Path(sequences_path)
        self.tok = LoanTokenizer.load(tokenizer_path)
        self.mode = mode
        self.label_maps = label_maps or {"default": {}, "cure": {}}
        self.patch_size = patch_size
        self.rng = random.Random(seed)

        # Load parquet
        table = pq.read_table(str(self.path))
        self.df = table.to_pandas()
        print(f"  Dataset: {len(self.df):,} loans, mode={mode}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        loan_id = str(row["loan_id"])
        seq_len = int(row["seq_len"])
        flat = list(row["seq_tokens"])

        tokens = np.array(flat, dtype=np.int64).reshape(MAX_SEQ_LEN, STEP_WIDTH)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        attn_mask[:seq_len] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._apply_patch_masking(tokens, seq_len)
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels": torch.from_numpy(labels),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }

        elif self.mode == "joint":
            # Next-step prediction: input is steps 0..T-2, target is steps 1..T-1
            input_ids = tokens.copy()
            # Targets: event tokens shifted by 1
            targets = np.full(MAX_SEQ_LEN, -100, dtype=np.int64)
            for t in range(min(seq_len - 1, MAX_SEQ_LEN - 1)):
                targets[t] = tokens[t + 1, EVENT_POS]
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels": torch.from_numpy(targets),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }

        else:  # finetune
            input_ids = tokens.copy()
            default_label = self.label_maps.get("default", {}).get(loan_id, 0)
            cure_label = self.label_maps.get("cure", {}).get(loan_id, 0)
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "default_label": torch.tensor(default_label, dtype=torch.long),
                "cure_label": torch.tensor(cure_label, dtype=torch.long),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }

    def _apply_patch_masking(
        self, tokens: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Mask 15% of patches (groups of patch_size steps).
        Within masked patches, replace event tokens with [MASK] ID.
        """
        input_ids = tokens.copy()
        labels = np.full(MAX_SEQ_LEN, -100, dtype=np.int64)

        n_patches = seq_len // self.patch_size
        if n_patches < 1:
            n_patches = 1

        for p in range(n_patches):
            if self.rng.random() < MASK_PROB:
                start = p * self.patch_size
                end = min(start + self.patch_size, seq_len)
                for t in range(start, end):
                    labels[t] = tokens[t, EVENT_POS]
                    input_ids[t, EVENT_POS] = self.tok.mask_id

        return input_ids, labels

    def oot_split(
        self, train_year_max: int = 2024
    ) -> tuple["_SubsetDataset", "_SubsetDataset"]:
        """Out-of-Time split with 80/20 fallback."""
        if "obs_year_max" in self.df.columns:
            train_mask = self.df["obs_year_max"] <= train_year_max
            test_mask = self.df["obs_year_min"] > train_year_max

            if test_mask.sum() > 0:
                train_df = self.df[train_mask].reset_index(drop=True)
                test_df = self.df[test_mask].reset_index(drop=True)
                print(f"  OOT split: train={len(train_df):,}, test={len(test_df):,}")
                return _SubsetDataset(self, train_df), _SubsetDataset(self, test_df)

        # Fallback: random 80/20
        n_train = int(len(self.df) * 0.8)
        shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = shuffled.iloc[:n_train].reset_index(drop=True)
        test_df = shuffled.iloc[n_train:].reset_index(drop=True)
        print(f"  Random split (no OOT data): train={len(train_df):,}, test={len(test_df):,}")
        return _SubsetDataset(self, train_df), _SubsetDataset(self, test_df)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    @staticmethod
    def load_label_maps(db_path: str | Path) -> dict[str, dict[str, int]]:
        """
        Load default and cure label maps from DuckDB.

        Returns:
            {'default': {loan_id: 0/1}, 'cure': {loan_id: 0/1}}
        """
        import duckdb

        db_path = Path(db_path)
        if not db_path.exists():
            print(f"  [WARNING] DB not found at {db_path}, using empty label maps")
            return {"default": {}, "cure": {}}

        con = duckdb.connect(str(db_path), read_only=True)

        # Default labels
        try:
            df_default = con.execute(
                "SELECT loan_id, MAX(default_in_3m) as label FROM gold_features GROUP BY loan_id"
            ).fetchdf()
            default_map = dict(zip(df_default["loan_id"].astype(str), df_default["label"].astype(int)))
        except Exception as e:
            print(f"  [WARNING] Could not load default labels: {e}")
            default_map = {}

        # Cure labels: check if a loan had a delinquent-to-performing transition
        try:
            cure_df = con.execute("""
                WITH transitions AS (
                    SELECT
                        loan_id,
                        arrears_bucket,
                        LAG(arrears_bucket) OVER (PARTITION BY loan_id ORDER BY reporting_date) AS prev_bucket
                    FROM dynamic_performance
                )
                SELECT
                    loan_id,
                    MAX(CASE
                        WHEN prev_bucket IN ('1-29 DPD','30-59 DPD','60-89 DPD','90+ DPD')
                        AND arrears_bucket = 'Performing' THEN 1
                        ELSE 0
                    END) AS had_cure
                FROM transitions
                GROUP BY loan_id
            """).fetchdf()
            cure_map = dict(zip(cure_df["loan_id"].astype(str), cure_df["had_cure"].astype(int)))
        except Exception as e:
            print(f"  [WARNING] Could not load cure labels: {e}")
            cure_map = {}

        con.close()
        print(f"  Labels: {sum(default_map.values())} defaults, {sum(cure_map.values())} cures")
        return {"default": default_map, "cure": cure_map}


class _SubsetDataset(Dataset):
    """Subset of CreditSequenceDataset with a filtered DataFrame."""

    def __init__(self, parent: CreditSequenceDataset, subset_df):
        self._parent = parent
        self.df = subset_df
        self.mode = parent.mode
        self.tok = parent.tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        loan_id = str(row["loan_id"])
        seq_len = int(row["seq_len"])
        flat = list(row["seq_tokens"])

        tokens = np.array(flat, dtype=np.int64).reshape(MAX_SEQ_LEN, STEP_WIDTH)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        attn_mask[:seq_len] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._parent._apply_patch_masking(tokens, seq_len)
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels": torch.from_numpy(labels),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }
        elif self.mode == "joint":
            input_ids = tokens.copy()
            targets = np.full(MAX_SEQ_LEN, -100, dtype=np.int64)
            for t in range(min(seq_len - 1, MAX_SEQ_LEN - 1)):
                targets[t] = tokens[t + 1, EVENT_POS]
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "labels": torch.from_numpy(targets),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }
        else:
            input_ids = tokens.copy()
            default_label = self._parent.label_maps.get("default", {}).get(loan_id, 0)
            cure_label = self._parent.label_maps.get("cure", {}).get(loan_id, 0)
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attn_mask),
                "default_label": torch.tensor(default_label, dtype=torch.long),
                "cure_label": torch.tensor(cure_label, dtype=torch.long),
                "loan_id": loan_id,
                "seq_len": seq_len,
            }

    @property
    def vocab_size(self):
        return self._parent.vocab_size


def collate_fn(batch: list[dict]) -> dict:
    """Universal collate function for all modes."""
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "loan_ids": [b["loan_id"] for b in batch],
    }

    if "labels" in batch[0]:
        result["labels"] = torch.stack([b["labels"] for b in batch])
    if "default_label" in batch[0]:
        result["default_label"] = torch.stack([b["default_label"] for b in batch])
        result["cure_label"] = torch.stack([b["cure_label"] for b in batch])

    return result
