"""
baseline_model.py — XGBoost Baseline Default Prediction
=========================================================
Queries gold_features from DuckDB (zero-copy via PyArrow),
performs an Out-of-Time split (2024 train / 2025 test),
trains an XGBoost classifier to predict default_in_3m,
and saves performance metrics + feature importances to JSON.

Usage:
    python baseline_model.py [--db PATH] [--output PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

WORKSPACE_DIR = Path(__file__).parent
DEFAULT_DB    = WORKSPACE_DIR / "credit_validate.db"
DEFAULT_OUT   = WORKSPACE_DIR / "model_results.json"

# Feature columns to use for training
# Mix of static and dynamic features; excludes keys and labels
FEATURE_COLS = [
    # Static numeric
    "origination_year", "original_balance", "legal_maturity_months",
    "oltomv_original", "original_market_value_at_origination",
    "loan_to_income", "payment_due_to_income_pct", "borrower_annual_income",
    "debtor_count", "construction_year",
    "primary_energy_demand_kwh_m2",
    # Static binary
    "interest_only_flag", "self_employed_flag", "buy_to_let_flag", "nhg_flag",
    # Dynamic numeric
    "current_balance", "current_interest_rate_pct", "remaining_term_months",
    "seasoning_months", "cltomv_current", "cltimv_current",
    "arrears_amount", "days_past_due",
    # Dynamic binary
    "default_crr_flag", "foreclosure_flag", "forbearance_flag", "restructuring_flag",
    # Derived momentum
    "balance_mom_1m",
]

CATEGORICAL_COLS = [
    "repayment_type", "rate_type", "borrower_type", "employment_status",
    "loan_purpose", "province", "economic_region_nuts3",
    "construction_year_bucket",  # ordinal
    "occupancy", "property_type", "property_usage",
    "arrears_bucket", "performing_status", "epc_label",
    "balance_bucket", "cltomv_current_bucket",
]

TARGET_COL = "default_in_3m"


def load_features(db_path: Path) -> tuple:
    """Load gold_features from DuckDB as PyArrow, convert to numpy/pandas."""
    import pandas as pd

    print("  Connecting to DuckDB ...")
    con = duckdb.connect(str(db_path), read_only=True)

    all_cols = FEATURE_COLS + CATEGORICAL_COLS + [TARGET_COL, "obs_year", "loan_id", "reporting_date"]
    # Only request columns that exist in the view
    available = [r[0] for r in con.execute("DESCRIBE gold_features").fetchall()]
    select_cols = [c for c in all_cols if c in available]

    print("  Querying gold_features (zero-copy Arrow) ...")
    t0 = time.perf_counter()
    query = con.execute(f"SELECT {', '.join(select_cols)} FROM gold_features")
    # fetch_arrow_table() is stable across DuckDB versions
    try:
        arrow_tbl = query.to_arrow_table()
        df = arrow_tbl.to_pandas()
    except AttributeError:
        df = query.fetchdf()
    con.close()
    elapsed = time.perf_counter() - t0
    print(f"  OK Loaded {len(df):,} rows x {len(df.columns)} cols in {elapsed:.2f}s")
    return df


def encode_categoricals(df, cat_cols: list) -> tuple:
    """Label-encode categorical columns present in the dataframe."""
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    present = [c for c in cat_cols if c in df.columns]
    for col in present:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("__NULL__"))
        encoders[col] = le
    return df, encoders


def train_xgboost(X_train, y_train, X_test, y_test, feature_names) -> dict:
    """Train XGBoost and return metrics + feature importances."""
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, roc_curve
    )

    print("\n  Training XGBoost classifier ...")
    print(f"    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"    Train positive rate: {y_train.mean():.4f}")
    print(f"    Test  positive rate: {y_test.mean():.4f}")

    # Class imbalance correction
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"    scale_pos_weight: {pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    train_time = time.perf_counter() - t0
    print(f"  OK Training complete in {train_time:.1f}s")

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict_proba(X_train)[:, 1]

    auc_roc_test  = roc_auc_score(y_test,  y_pred_proba)
    auc_roc_train = roc_auc_score(y_train, y_pred_train)
    gini_test     = 2 * auc_roc_test  - 1
    gini_train    = 2 * auc_roc_train - 1
    avg_prec      = average_precision_score(y_test, y_pred_proba)

    print(f"\n  Test  AUC-ROC : {auc_roc_test:.4f}  (Gini: {gini_test:.4f})")
    print(f"  Train AUC-ROC : {auc_roc_train:.4f}  (Gini: {gini_train:.4f})")
    print(f"  Avg Precision  : {avg_prec:.4f}")

    # ROC curve (100 points for frontend)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    idx_pr = np.linspace(0, len(prec) - 1, min(100, len(prec)), dtype=int)

    # Feature importances
    importance = model.feature_importances_
    feat_imp = sorted(
        [{"feature": name, "importance": float(imp)}
         for name, imp in zip(feature_names, importance)],
        key=lambda x: x["importance"], reverse=True
    )[:20]  # top 20

    # Evals (training curves)
    evals = model.evals_result()
    train_curve = evals["validation_0"]["auc"]
    test_curve  = evals["validation_1"]["auc"]

    return {
        "auc_roc_test":    round(auc_roc_test,  4),
        "auc_roc_train":   round(auc_roc_train, 4),
        "gini_test":       round(gini_test,     4),
        "gini_train":      round(gini_train,    4),
        "avg_precision":   round(avg_prec,      4),
        "train_seconds":   round(train_time, 2),
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[idx]],
            "tpr": [round(float(v), 4) for v in tpr[idx]],
        },
        "pr_curve": {
            "precision": [round(float(v), 4) for v in prec[idx_pr]],
            "recall":    [round(float(v), 4) for v in rec[idx_pr]],
        },
        "feature_importances": feat_imp,
        "training_curve": {
            "train_auc": [round(v, 4) for v in train_curve],
            "test_auc":  [round(v, 4) for v in test_curve],
        },
        "train_rows": int(len(X_train)),
        "test_rows":  int(len(X_test)),
        "train_default_rate": round(float(y_train.mean()), 4),
        "test_default_rate":  round(float(y_test.mean()),  4),
    }


def main():
    parser = argparse.ArgumentParser(description="CFM Baseline XGBoost Model")
    parser.add_argument("--db",     type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("=" * 60)
    print("  Credit Foundation Model — XGBoost Baseline")
    print("=" * 60)
    print(f"  Database: {args.db}")

    if not args.db.exists():
        print("ERROR: Database not found. Run ingest_bronze.py first.")
        return 1

    # -- Load features --------------------------------------------------
    df = load_features(args.db)

    # -- Encode categoricals --------------------------------------------
    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    df, _ = encode_categoricals(df, CATEGORICAL_COLS)
    present_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    all_features = present_feats + present_cats

    # -- Out-of-Time split --------------------------------------------─
    # Train on 2024, Test on 2025 (prevents temporal data leakage)
    train_mask = df["obs_year"] <= 2024
    test_mask  = df["obs_year"] >= 2025

    if test_mask.sum() == 0:
        # Fallback: last 20% by date
        sorted_df  = df.sort_values("reporting_date")
        split_idx  = int(len(sorted_df) * 0.8)
        train_mask = df.index.isin(sorted_df.index[:split_idx])
        test_mask  = df.index.isin(sorted_df.index[split_idx:])
        print("  [WARNING] No 2025 data found. Using 80/20 temporal split instead.")

    X_train = df[train_mask][all_features].fillna(-999).values
    y_train = df[train_mask][TARGET_COL].values.astype(int)
    X_test  = df[test_mask][all_features].fillna(-999).values
    y_test  = df[test_mask][TARGET_COL].values.astype(int)

    # -- Train ----------------------------------------------------------
    metrics = train_xgboost(X_train, y_train, X_test, y_test, all_features)
    metrics["features_used"] = all_features
    metrics["split_method"]  = "Out-of-Time (train≤2024 / test≥2025)"

    # -- Save results --------------------------------------------------─
    args.output.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  Model Results saved -> {args.output.name}")
    print(f"  Test AUC-ROC : {metrics['auc_roc_test']:.4f}")
    print(f"  Test Gini    : {metrics['gini_test']:.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
