"""
build_sequences.py — Tokenize and Construct Loan Event Sequences
================================================================
Reads dynamic_performance from DuckDB, tokenizes each loan's monthly
history using LoanTokenizer, and writes a compact sequence store to
sequences.parquet.

Each row in sequences.parquet represents one complete loan history:
  loan_id       - string identifier
  seq_tokens    - flat int32 array: shape (MAX_SEQ_LEN * STEP_WIDTH,)
  seq_len       - int: number of non-padded time steps (1..24)
  event_seq     - JSON string: list of lifecycle event token names
  final_state   - string: last observed lifecycle event token
  had_cure      - bool: any DPD*->PERF cure observed
  had_default   - bool: any DFLT state observed
  obs_year_min  - int: first year of observations (for OOT split)
  obs_year_max  - int: last year of observations

Usage:
    python build_sequences.py [--db PATH] [--out PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tokenizer import (
    LoanTokenizer, MAX_SEQ_LEN, STEP_WIDTH,
    ARREARS_TO_EVENT, LIFECYCLE_TOKENS
)

WORKSPACE_DIR   = Path(__file__).parent
DEFAULT_DB      = WORKSPACE_DIR.parent / "pipelines" / "credit_validate.db"
DEFAULT_OUT     = WORKSPACE_DIR.parent / "tokenize-sequences-result" / "sequences.parquet"
TOKENIZER_PATH  = WORKSPACE_DIR.parent / "tokenize-sequences-result" / "tokenizer.json"

# Fetch columns needed for tokenization
FETCH_COLS = [
    "loan_id", "reporting_date",
    "arrears_bucket",
    "cltomv_current", "current_interest_rate_pct",
    "current_balance", "days_past_due",
    "default_crr_flag",
]

PAD_STEP = [0] * STEP_WIDTH   # [PAD] repeated for all step positions


def fit_and_save_tokenizer(con: duckdb.DuckDBPyConnection) -> LoanTokenizer:
    """Fit the tokenizer on the full dynamic_performance table."""
    tok = LoanTokenizer()
    tok.fit(con)
    tok.save(TOKENIZER_PATH)
    return tok


def build_sequences(
    con: duckdb.DuckDBPyConnection,
    tok: LoanTokenizer,
) -> tuple[list[dict], dict]:
    """
    Read all loan histories from DuckDB and tokenize them.
    Returns (list_of_records, stats_dict).
    """
    print("\n  Reading dynamic_performance ordered by (loan_id, reporting_date) ...")
    t0 = time.perf_counter()

    cols = ", ".join(f'"{c}"' for c in FETCH_COLS)
    df = con.execute(
        f"SELECT {cols} FROM dynamic_performance ORDER BY loan_id, reporting_date"
    ).fetchdf()

    elapsed_query = time.perf_counter() - t0
    print(f"  OK Loaded {len(df):,} rows in {elapsed_query:.2f}s")

    # -- Group by loan_id and build sequences ------------------------
    print("  Tokenizing sequences ...")
    t1 = time.perf_counter()

    records = []
    total_loans = 0
    total_cures = 0
    total_defaults = 0
    state_trans_counts: dict[tuple, int] = {}  # (from_tok, to_tok) -> count
    event_freq: dict[str, int] = {t: 0 for t in LIFECYCLE_TOKENS}
    seq_len_hist: list[int] = []

    grouped = df.groupby("loan_id", sort=False)

    for loan_id, group in grouped:
        group = group.sort_values("reporting_date").reset_index(drop=True)
        n_steps = len(group)

        token_steps: list[list[int]] = []
        event_names: list[str] = []
        had_cure    = False
        had_default = False
        prev_arrears: str | None = None

        for _, row in group.iterrows():
            row_dict = row.to_dict()
            arrears  = str(row_dict.get("arrears_bucket", "Performing"))

            # Cure detection: DPD* -> Performing
            if prev_arrears is not None and tok.is_cure(prev_arrears, arrears):
                had_cure = True

            # State transition matrix
            if prev_arrears is not None:
                from_tok = ARREARS_TO_EVENT.get(prev_arrears, "PERF")
                to_tok   = ARREARS_TO_EVENT.get(arrears, "PERF")
                state_trans_counts[(from_tok, to_tok)] = \
                    state_trans_counts.get((from_tok, to_tok), 0) + 1

            # Default detection
            if row_dict.get("default_crr_flag") is True:
                had_default = True

            # Encode this monthly step
            step = tok.encode_step(row_dict)
            token_steps.append(step)

            ev_name = ARREARS_TO_EVENT.get(arrears, "PERF")
            event_names.append(ev_name)
            event_freq[ev_name] = event_freq.get(ev_name, 0) + 1

            prev_arrears = arrears

        # Pad / truncate to MAX_SEQ_LEN
        seq_len    = min(n_steps, MAX_SEQ_LEN)
        padded     = token_steps[:MAX_SEQ_LEN]
        while len(padded) < MAX_SEQ_LEN:
            padded.append(list(PAD_STEP))

        # Add BOS at front, shift everything (keep total length=MAX_SEQ_LEN)
        # BOS replaces padding of first slot only when seq_len >= MAX_SEQ_LEN
        # For shorter sequences, BOS is prepended and EOS appended inside pad
        # Simple: store raw padded sequence, BOS/EOS added in DataLoader

        flat_tokens = []
        for step in padded:
            flat_tokens.extend(step)

        final_state = event_names[-1] if event_names else "PERF"
        obs_years = [int(str(d)[:4]) for d in group["reporting_date"]]

        records.append({
            "loan_id":      loan_id,
            "seq_tokens":   flat_tokens,             # list of int
            "seq_len":      seq_len,
            "event_seq":    json.dumps(event_names[:MAX_SEQ_LEN]),
            "final_state":  final_state,
            "had_cure":     had_cure,
            "had_default":  had_default,
            "obs_year_min": min(obs_years),
            "obs_year_max": max(obs_years),
        })

        total_loans   += 1
        seq_len_hist.append(seq_len)
        if had_cure:    total_cures    += 1
        if had_default: total_defaults += 1

    elapsed_tok = time.perf_counter() - t1
    print(f"  OK Tokenized {total_loans:,} loan sequences in {elapsed_tok:.2f}s")
    print(f"    Loans with cure events  : {total_cures:,} ({100*total_cures/total_loans:.2f}%)")
    print(f"    Loans with default      : {total_defaults:,} ({100*total_defaults/total_loans:.2f}%)")
    print(f"    Avg sequence length     : {np.mean(seq_len_hist):.1f} steps")

    stats = {
        "total_loans":          total_loans,
        "total_cures":          total_cures,
        "total_defaults":       total_defaults,
        "cure_rate_pct":        round(100 * total_cures / total_loans, 3),
        "default_rate_pct":     round(100 * total_defaults / total_loans, 3),
        "avg_seq_len":          round(float(np.mean(seq_len_hist)), 2),
        "min_seq_len":          int(np.min(seq_len_hist)),
        "max_seq_len":          int(np.max(seq_len_hist)),
        "seq_len_distribution": {
            str(l): int(seq_len_hist.count(l))
            for l in sorted(set(seq_len_hist))
        },
        "event_frequency":      event_freq,
        "state_transitions":    {
            f"{k[0]}->{k[1]}": v
            for k, v in sorted(state_trans_counts.items(),
                                key=lambda x: -x[1])
        },
        "vocab_size":           tok.vocab_size,
        "step_width":           STEP_WIDTH,
        "max_seq_len":          MAX_SEQ_LEN,
        "query_seconds":        round(elapsed_query, 3),
        "tokenize_seconds":     round(elapsed_tok, 3),
    }

    return records, stats


def build_transition_matrix(stats: dict) -> dict:
    """
    Build a normalized 8x8 state transition probability matrix.
    Returns dict with 'states' list and 'matrix' 2D list.
    """
    states = ["PERF", "DPD1", "DPD2", "DPD3", "DPD4", "DFLT", "CHOF", "RDMD"]
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((n, n), dtype=float)

    for key, count in stats.get("state_transitions", {}).items():
        parts = key.split("->")
        if len(parts) == 2 and parts[0] in idx and parts[1] in idx:
            i, j = idx[parts[0]], idx[parts[1]]
            matrix[i, j] = count

    # Row-normalize (so each row sums to 1 where there are observations)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0   # avoid div-by-zero for unobserved states
    matrix_norm = (matrix / row_sums).tolist()

    return {
        "states": states,
        "matrix": matrix_norm,
        "raw_counts": matrix.tolist(),
    }


def write_parquet(records: list[dict], out_path: Path) -> None:
    """Write sequence records to Parquet using PyArrow."""
    print(f"\n  Writing {len(records):,} sequences to {out_path.name} ...")
    t0 = time.perf_counter()

    schema = pa.schema([
        pa.field("loan_id",      pa.string()),
        pa.field("seq_tokens",   pa.list_(pa.int32())),
        pa.field("seq_len",      pa.int32()),
        pa.field("event_seq",    pa.string()),
        pa.field("final_state",  pa.string()),
        pa.field("had_cure",     pa.bool_()),
        pa.field("had_default",  pa.bool_()),
        pa.field("obs_year_min", pa.int32()),
        pa.field("obs_year_max", pa.int32()),
    ])

    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, str(out_path), compression="snappy")

    size_mb = out_path.stat().st_size / 1_048_576
    elapsed = time.perf_counter() - t0
    print(f"  OK Wrote {len(records):,} rows ({size_mb:.1f} MB) in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="CFM Step 2: Build Loan Sequences")
    parser.add_argument("--db",  type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("=" * 60)
    print("  Credit Foundation Model — Tokenize & Build Sequences")
    print("=" * 60)
    print(f"  Database   : {args.db}")
    print(f"  Output     : {args.out}")

    if not args.db.exists():
        print("ERROR: Database not found. Run ingest_bronze.py first.")
        return 1

    con = duckdb.connect(str(args.db), read_only=True)

    # Step A: Fit tokenizer
    print("\n-- Step A: Fit Tokenizer --")
    tok = fit_and_save_tokenizer(con)

    # Step B: Build sequences
    print("\n-- Step B: Build Sequences --")
    records, stats = build_sequences(con, tok)
    con.close()

    # Step C: Build transition matrix
    tm = build_transition_matrix(stats)
    stats["transition_matrix"] = tm

    # Step D: Write Parquet
    write_parquet(records, args.out)

    # Step E: Save stats
    stats_path = WORKSPACE_DIR.parent / "tokenize-sequences-result" / "sequence_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # Step F: Save tokenizer summary
    tok_summary = tok.summary()
    tok_summary_path = WORKSPACE_DIR.parent / "tokenize-sequences-result" / "tokenizer_summary.json"
    tok_summary_path.write_text(json.dumps(tok_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  Sequence Construction Complete")
    print("=" * 60)
    print(f"  Unique loans      : {stats['total_loans']:>10,}")
    print(f"  Vocabulary size   : {stats['vocab_size']:>10,} tokens")
    print(f"  Avg sequence len  : {stats['avg_seq_len']:>10.1f} months")
    print(f"  Cure rate         : {stats['cure_rate_pct']:>10.2f}%")
    print(f"  Default rate      : {stats['default_rate_pct']:>10.2f}%")
    print(f"  sequences.parquet : {args.out.stat().st_size/1_048_576:>10.1f} MB")
    print(f"  Stats saved       -> {stats_path.name}")
    print(f"  Tokenizer saved   -> {TOKENIZER_PATH.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
