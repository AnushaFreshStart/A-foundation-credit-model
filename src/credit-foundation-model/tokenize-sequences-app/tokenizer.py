"""
tokenizer.py — Loan Event Tokenizer & Vocabulary Registry
==========================================================
Builds a three-tier vocabulary for credit sequence tokenization:
  Tier 1: Special tokens  [PAD] [UNK] [BOS] [EOS] [MASK]
  Tier 2: Lifecycle event tokens  PERF DPD1 DPD2 DPD3 DPD4 DFLT CHOF RDMD
  Tier 3: Quantile-binned continuous features
          CLTV_Qn / RATE_Qn / BAL_Qn / DPD_Qn  (10 bins each)

Usage:
    tok = LoanTokenizer()
    tok.fit(duckdb_connection)
    ids = tok.encode_step(row_dict)     # -> list[int] of length STEP_WIDTH
    tok.save("tokenizer.json")
    tok2 = LoanTokenizer.load("tokenizer.json")
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# -- Constants ------------------------------------------------
MAX_SEQ_LEN = 24      # 24 monthly cutoffs
N_BINS      = 10      # quantile bins for continuous features
STEP_WIDTH  = 5       # tokens per time-step: [event, cltv, rate, bal, dpd]

# Lifecycle state mapping from arrears_bucket
ARREARS_TO_EVENT = {
    "Performing":  "PERF",
    "1-29 DPD":    "DPD1",
    "30-59 DPD":   "DPD2",
    "60-89 DPD":   "DPD3",
    "90+ DPD":     "DPD4",
    "Defaulted":   "DFLT",
    "Charged-Off": "CHOF",
    "Redeemed":    "RDMD",
}

# The special "cure" pseudo-event (DPD* -> PERF) is handled by build_sequences.py
# but stored here for reference
CURE_EVENT = "CURE"

# Tier-1 special tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

# Tier-2 lifecycle event tokens
LIFECYCLE_TOKENS = ["PERF", "DPD1", "DPD2", "DPD3", "DPD4", "DFLT", "CHOF", "RDMD"]

# Tier-3 continuous feature names
CONTINUOUS_FEATURES = ["CLTV", "RATE", "BAL", "DPD"]


class VocabRegistry:
    """Maps token names to integer IDs and back."""

    def __init__(self):
        self._tok2id: dict[str, int] = {}
        self._id2tok: dict[int, str] = {}

    def add(self, token: str) -> int:
        if token not in self._tok2id:
            idx = len(self._tok2id)
            self._tok2id[token] = idx
            self._id2tok[idx] = token
        return self._tok2id[token]

    def __getitem__(self, token: str) -> int:
        return self._tok2id.get(token, self._tok2id["[UNK]"])

    def decode(self, idx: int) -> str:
        return self._id2tok.get(idx, "[UNK]")

    def __len__(self) -> int:
        return len(self._tok2id)

    def to_dict(self) -> dict:
        return dict(self._tok2id)

    @classmethod
    def from_dict(cls, d: dict) -> "VocabRegistry":
        reg = cls()
        for tok, idx in sorted(d.items(), key=lambda x: x[1]):
            reg._tok2id[tok] = idx
            reg._id2tok[idx] = tok
        return reg


class LoanTokenizer:
    """
    Fits on the dynamic_performance table and encodes monthly snapshots
    into sequences of integer token IDs.
    """

    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3
    MASK_ID = 4

    def __init__(self):
        self.vocab    = VocabRegistry()
        self.bin_edges: dict[str, list[float]] = {}
        self._fitted  = False

        # Build vocabulary immediately (tiers 1+2 are static)
        self._build_static_vocab()

    def _build_static_vocab(self):
        """Register special and lifecycle tokens."""
        for tok in SPECIAL_TOKENS:
            self.vocab.add(tok)
        for tok in LIFECYCLE_TOKENS:
            self.vocab.add(tok)

    def fit(self, con) -> "LoanTokenizer":
        """
        Compute quantile bin edges for each continuous feature
        by scanning the dynamic_performance table.
        """
        print("  Fitting tokenizer on dynamic_performance ...")

        feature_queries = {
            "CLTV": "SELECT cltomv_current           FROM dynamic_performance WHERE cltomv_current IS NOT NULL AND cltomv_current > 0 AND cltomv_current < 200",
            "RATE": "SELECT current_interest_rate_pct FROM dynamic_performance WHERE current_interest_rate_pct IS NOT NULL AND current_interest_rate_pct > 0",
            "BAL":  "SELECT current_balance           FROM dynamic_performance WHERE current_balance IS NOT NULL AND current_balance > 0",
            "DPD":  "SELECT days_past_due             FROM dynamic_performance WHERE days_past_due IS NOT NULL AND days_past_due >= 0",
        }

        for feat, sql in feature_queries.items():
            values = con.execute(sql).fetchnumpy()[sql.split("SELECT")[1].split("FROM")[0].strip()]
            values = values[~np.isnan(values)]
            quantiles = np.percentile(values, np.linspace(0, 100, N_BINS + 1)).tolist()
            # De-duplicate edges (common when many zeros)
            edges = sorted(set(quantiles))
            if len(edges) < 2:
                edges = [float(values.min()), float(values.max()) + 1e-9]
            self.bin_edges[feat] = edges

            # Register bin tokens
            for i in range(N_BINS):
                tok = f"{feat}_Q{i}"
                self.vocab.add(tok)

            print(f"    {feat}: {len(edges)-1} effective bins, range [{values.min():.2f}, {values.max():.2f}]")

        self._fitted = True
        print(f"  OK Vocabulary size: {len(self.vocab)} tokens")
        return self

    def _quantize(self, feat: str, value: float | None) -> int:
        """Map a scalar value to its bin token ID."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self.vocab["[UNK]"]
        edges = self.bin_edges.get(feat, [])
        if not edges:
            return self.vocab["[UNK]"]
        # np.searchsorted: find bin index (0..N_BINS-1)
        idx = int(np.searchsorted(edges[1:-1], value, side="right"))
        idx = min(idx, N_BINS - 1)
        return self.vocab[f"{feat}_Q{idx}"]

    def encode_step(self, row: dict[str, Any]) -> list[int]:
        """
        Encode one monthly snapshot dict into STEP_WIDTH token IDs.

        Returns: [event_id, cltv_bin_id, rate_bin_id, bal_bin_id, dpd_bin_id]
        """
        # Tier-2: lifecycle event
        arrears = str(row.get("arrears_bucket", "Performing"))
        event_tok = ARREARS_TO_EVENT.get(arrears, "PERF")
        event_id = self.vocab[event_tok]

        # Tier-3: continuous bins
        cltv_id = self._quantize("CLTV", row.get("cltomv_current"))
        rate_id = self._quantize("RATE", row.get("current_interest_rate_pct"))
        bal_id  = self._quantize("BAL",  row.get("current_balance"))
        dpd_id  = self._quantize("DPD",  row.get("days_past_due"))

        return [event_id, cltv_id, rate_id, bal_id, dpd_id]

    def event_token(self, arrears_bucket: str) -> str:
        """Return the string event token for an arrears_bucket value."""
        return ARREARS_TO_EVENT.get(arrears_bucket, "PERF")

    def is_cure(self, prev_arrears: str, curr_arrears: str) -> bool:
        """True if this step represents a cure transition."""
        was_delinquent = prev_arrears in ("1-29 DPD", "30-59 DPD", "60-89 DPD", "90+ DPD")
        now_performing = curr_arrears == "Performing"
        return was_delinquent and now_performing

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    @property
    def mask_id(self) -> int:
        return self.MASK_ID

    def save(self, path: str | Path) -> None:
        """Serialize tokenizer to JSON."""
        path = Path(path)
        data = {
            "vocab":      self.vocab.to_dict(),
            "bin_edges":  self.bin_edges,
            "n_bins":     N_BINS,
            "step_width": STEP_WIDTH,
            "max_seq_len": MAX_SEQ_LEN,
            "version":    "1.0",
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  OK Tokenizer saved -> {path.name}  (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str | Path) -> "LoanTokenizer":
        """Deserialize tokenizer from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.vocab     = VocabRegistry.from_dict(data["vocab"])
        tok.bin_edges = data["bin_edges"]
        tok._fitted   = True
        return tok

    def summary(self) -> dict:
        """Return a summary dict for dashboard display."""
        tier1 = {t: self.vocab[t] for t in SPECIAL_TOKENS}
        tier2 = {t: self.vocab[t] for t in LIFECYCLE_TOKENS}
        tier3 = {}
        for feat in CONTINUOUS_FEATURES:
            tier3[feat] = {
                f"{feat}_Q{i}": self.vocab[f"{feat}_Q{i}"]
                for i in range(N_BINS)
                if f"{feat}_Q{i}" in self.vocab.to_dict()
            }
        return {
            "vocab_size":   self.vocab_size,
            "step_width":   STEP_WIDTH,
            "max_seq_len":  MAX_SEQ_LEN,
            "n_bins":       N_BINS,
            "tiers": {
                "special":    tier1,
                "lifecycle":  tier2,
                "continuous": tier3,
            },
            "bin_edges": self.bin_edges,
        }
