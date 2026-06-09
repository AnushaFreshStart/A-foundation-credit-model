"""
sequence_dataset.py — PyTorch Dataset for Transformer Pre-training & Fine-tuning
=================================================================================
Provides two modes:
  "pretrain"  -> Masked Token Modeling (MTM): mask 15% of event tokens at random
  "finetune"  -> Supervised classification: return (sequence, default_in_3m label)

Usage:
    from sequence_dataset import LoanSequenceDataset, LoanDataCollator

    dataset = LoanSequenceDataset("sequences.parquet", "tokenizer.json", mode="pretrain")
    train_ds, val_ds = dataset.oot_split(train_year_max=2024)

    loader = DataLoader(train_ds, batch_size=64, collate_fn=LoanDataCollator())
    for batch in loader:
        input_ids, attention_mask, labels = batch
"""

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import LoanTokenizer, MAX_SEQ_LEN, STEP_WIDTH

# -- Hyper-parameters ----------------------------------------─
MASK_PROB   = 0.15   # fraction of event tokens to mask in pretrain mode
EVENT_POS   = 0      # position of event token within each STEP_WIDTH block


class LoanSequenceDataset(Dataset):
    """
    PyTorch Dataset over sequences.parquet.

    Each item is a dict:
      "input_ids":      LongTensor [MAX_SEQ_LEN, STEP_WIDTH]
      "attention_mask": LongTensor [MAX_SEQ_LEN]  — 1=real token, 0=pad
      "labels":
        pretrain mode -> LongTensor [MAX_SEQ_LEN]  (event token id or -100 if unmasked)
        finetune mode -> scalar LongTensor          (0/1 default_in_3m)
      "loan_id":        str
      "seq_len":        int
    """

    def __init__(
        self,
        sequences_path: str | Path,
        tokenizer_path: str | Path,
        mode: Literal["pretrain", "finetune"] = "pretrain",
        label_map: dict[str, int] | None = None,
        seed: int = 42,
    ):
        self.path      = Path(sequences_path)
        self.tok       = LoanTokenizer.load(tokenizer_path)
        self.mode      = mode
        self.label_map = label_map or {}   # loan_id -> 0/1 label for finetune
        self.rng       = random.Random(seed)

        # Load parquet into memory (small: 10k loans x 24 x 5 ints)
        table = pq.read_table(str(self.path))
        self.df = table.to_pandas()
        print(f"  Loaded {len(self.df):,} loan sequences from {self.path.name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        loan_id  = str(row["loan_id"])
        seq_len  = int(row["seq_len"])
        flat     = list(row["seq_tokens"])  # MAX_SEQ_LEN * STEP_WIDTH ints

        # Reshape to [MAX_SEQ_LEN, STEP_WIDTH]
        tokens = np.array(flat, dtype=np.int64).reshape(MAX_SEQ_LEN, STEP_WIDTH)

        # Attention mask: 1 for real steps, 0 for padded
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        attn_mask[:seq_len] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._apply_mtm(tokens, seq_len)
        else:
            input_ids = tokens.copy()
            # finetune label from label_map (built externally from gold_features)
            label_val = self.label_map.get(loan_id, 0)
            labels    = np.array(label_val, dtype=np.int64)

        return {
            "input_ids":      torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn_mask),
            "labels":         torch.from_numpy(labels) if isinstance(labels, np.ndarray) and labels.ndim > 0
                              else torch.tensor(labels, dtype=torch.long),
            "loan_id":        loan_id,
            "seq_len":        seq_len,
        }

    def _apply_mtm(
        self,
        tokens: np.ndarray,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Masked Token Modeling: for each real step, independently mask the
        event token (position 0 in the step) with probability MASK_PROB.

        Returns:
          input_ids: modified token array with [MASK] at masked positions
          labels:    original event token ids at masked positions, -100 elsewhere
        """
        input_ids = tokens.copy()
        labels    = np.full(MAX_SEQ_LEN, -100, dtype=np.int64)  # ignore index

        for t in range(seq_len):
            if self.rng.random() < MASK_PROB:
                original_event_id    = tokens[t, EVENT_POS]
                labels[t]            = original_event_id
                input_ids[t, EVENT_POS] = self.tok.mask_id

        return input_ids, labels

    def oot_split(
        self,
        train_year_max: int = 2024,
    ) -> tuple["LoanSequenceDataset", "LoanSequenceDataset"]:
        """
        Out-of-Time split: train on sequences whose obs_year_max ≤ train_year_max,
        test on sequences whose obs_year_min > train_year_max.
        Returns (train_dataset, test_dataset).
        """
        train_mask = self.df["obs_year_max"] <= train_year_max
        test_mask  = self.df["obs_year_min"] > train_year_max

        # If no OOT data (e.g. single-year datasets), fall back to 80/20
        if test_mask.sum() == 0:
            n_train  = int(len(self.df) * 0.8)
            shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            train_df = shuffled.iloc[:n_train]
            test_df  = shuffled.iloc[n_train:]
        else:
            train_df = self.df[train_mask].reset_index(drop=True)
            test_df  = self.df[test_mask].reset_index(drop=True)

        train_ds = _SubsetDataset(self, train_df)
        test_ds  = _SubsetDataset(self, test_df)
        return train_ds, test_ds

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    @property
    def pad_id(self) -> int:
        return self.tok.pad_id

    @staticmethod
    def load_label_map(gold_features_path: str | Path | None = None,
                       con=None) -> dict[str, int]:
        """
        Build a label_map: loan_id -> max(default_in_3m) from gold_features.
        Accepts either a DuckDB connection or a pre-exported Parquet path.
        """
        if con is not None:
            df = con.execute(
                "SELECT loan_id, MAX(default_in_3m) as label FROM gold_features GROUP BY loan_id"
            ).fetchdf()
            return dict(zip(df["loan_id"], df["label"].astype(int)))
        if gold_features_path is not None:
            import pandas as pd
            df = pd.read_parquet(gold_features_path)
            return dict(zip(df["loan_id"], df["default_in_3m"].astype(int)))
        return {}


class _SubsetDataset(Dataset):
    """Helper: subset of a LoanSequenceDataset using a pre-filtered DataFrame."""

    def __init__(self, parent: LoanSequenceDataset, subset_df):
        self._parent = parent
        self.df      = subset_df
        self.mode    = parent.mode
        self.tok     = parent.tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        loan_id  = str(row["loan_id"])
        seq_len  = int(row["seq_len"])
        flat     = list(row["seq_tokens"])

        tokens    = np.array(flat, dtype=np.int64).reshape(MAX_SEQ_LEN, STEP_WIDTH)
        attn_mask = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
        attn_mask[:seq_len] = 1

        if self.mode == "pretrain":
            input_ids, labels = self._parent._apply_mtm(tokens, seq_len)
        else:
            input_ids = tokens.copy()
            label_val = self._parent.label_map.get(loan_id, 0)
            labels    = np.array(label_val, dtype=np.int64)

        return {
            "input_ids":      torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn_mask),
            "labels":         torch.from_numpy(labels) if isinstance(labels, np.ndarray) and labels.ndim > 0
                              else torch.tensor(labels, dtype=torch.long),
            "loan_id":        loan_id,
            "seq_len":        seq_len,
        }

    @property
    def vocab_size(self):
        return self._parent.vocab_size


class LoanDataCollator:
    """
    Collate function for DataLoader.
    Stacks tensors, returns dict with:
      input_ids:      [B, MAX_SEQ_LEN, STEP_WIDTH]
      attention_mask: [B, MAX_SEQ_LEN]
      labels:         [B, MAX_SEQ_LEN] (pretrain) or [B] (finetune)
    """

    def __call__(self, batch: list[dict]) -> dict:
        input_ids      = torch.stack([b["input_ids"]      for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        # labels may be 1D (finetune) or 1D-per-step (pretrain)
        labels_list = [b["labels"] for b in batch]
        if labels_list[0].ndim == 0:
            labels = torch.stack(labels_list)
        else:
            labels = torch.stack(labels_list)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "loan_ids":       [b["loan_id"] for b in batch],
        }


# -- Smoke test ----------------------------------------------─
if __name__ == "__main__":
    import sys
    seq_path = Path(__file__).parent / "sequences.parquet"
    tok_path = Path(__file__).parent / "tokenizer.json"

    if not seq_path.exists():
        print("Run build_sequences.py first to generate sequences.parquet")
        sys.exit(1)

    print("Loading dataset in pretrain mode ...")
    ds = LoanSequenceDataset(seq_path, tok_path, mode="pretrain")
    train_ds, test_ds = ds.oot_split(train_year_max=2024)
    print(f"  Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")

    loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=LoanDataCollator(),
    )
    batch = next(iter(loader))
    print(f"  Batch input_ids shape : {batch['input_ids'].shape}")
    print(f"  Batch attn_mask shape : {batch['attention_mask'].shape}")
    print(f"  Batch labels shape    : {batch['labels'].shape}")
    print(f"  Masked tokens in batch: {(batch['labels'] != -100).sum().item()}")
    print("OK Dataset smoke test PASSED")
