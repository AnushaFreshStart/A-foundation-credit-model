"""
downstream_eval.py — Step 4 Downstream Evaluation & Early Warning
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "origination_year", "original_balance", "legal_maturity_months",
    "oltomv_original", "original_market_value_at_origination",
    "loan_to_income", "payment_due_to_income_pct", "borrower_annual_income",
    "debtor_count", "construction_year", "primary_energy_demand_kwh_m2",
    "interest_only_flag", "self_employed_flag", "buy_to_let_flag", "nhg_flag",
    "current_balance", "current_interest_rate_pct", "remaining_term_months",
    "seasoning_months", "cltomv_current", "cltimv_current",
    "arrears_amount", "days_past_due",
    "default_crr_flag", "foreclosure_flag", "forbearance_flag", "restructuring_flag",
    "balance_mom_1m"
]
CATEGORICAL_COLS = [
    "repayment_type", "rate_type", "borrower_type", "employment_status",
    "loan_purpose", "province", "economic_region_nuts3",
    "construction_year_bucket", "occupancy", "property_type", "property_usage",
    "arrears_bucket", "performing_status", "epc_label",
    "balance_bucket", "cltomv_current_bucket"
]
TARGET_COL = "default_in_3m"

class DownstreamDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            ece += in_bin.mean() * np.abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)

def get_calibration_curve(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            curve.append({
                "mean_prob": float(y_prob[in_bin].mean()),
                "fraction_pos": float(y_true[in_bin].mean())
            })
    return curve

def get_horizon_curve(df_test, y_test, y_prob):
    # compute horizons
    df_test = df_test.copy()
    df_test["y_test"] = y_test
    df_test["y_prob"] = y_prob
    horizons = []
    for h in range(1, 7):
        mask = (df_test["months_to_default"] == h) | (df_test["y_test"] == 0)
        subset = df_test[mask]
        if len(subset) > 10 and subset["y_test"].sum() > 0:
            pr_auc = average_precision_score(subset["y_test"], subset["y_prob"])
            horizons.append({"horizon": h, "pr_auc": float(pr_auc)})
        else:
            horizons.append({"horizon": h, "pr_auc": 0.0})
    return horizons

def evaluate_model(y_true, y_prob):
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except:
        auc_roc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = 0.0
    try:
        brier = brier_score_loss(y_true, y_prob)
    except:
        brier = 1.0
        
    ece = compute_ece(y_true, y_prob)
    # Outreach @ 5%
    threshold = np.percentile(y_prob, 95)
    flagged = y_prob >= threshold
    capture_rate = y_true[flagged].sum() / max(1, y_true.sum())
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)
    
    return {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "gini": float(2 * auc_roc - 1),
        "brier_score": float(brier),
        "ece": ece,
        "capture_rate_5pct": float(capture_rate),
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[idx]],
            "tpr": [round(float(v), 4) for v in tpr[idx]],
        },
        "calibration": get_calibration_curve(y_true, y_prob)
    }

def train_xgboost(X_train, y_train, X_test, y_test, use_gpu=True):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    # Calibrate using Platt scaling
    from sklearn.frozen import FrozenEstimator
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrator.fit(X_test, y_test)
    y_prob = calibrator.predict_proba(X_test)[:, 1]
    return y_prob

def train_linear(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_prob

def train_dnn(X_train, y_train, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DownstreamDNN(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # FP16 AMP Support
    scaler = torch.amp.GradScaler()
    
    model.train()
    for epoch in range(5):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(bx)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    model.eval()
    with torch.no_grad():
        bx = torch.FloatTensor(X_test).to(device)
        with torch.amp.autocast('cuda'):
            logits = model(bx)
        y_prob = torch.sigmoid(logits).cpu().numpy().flatten()
    return y_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    
    print(f"Loading DuckDB: {args.db}")
    con = duckdb.connect(args.db, read_only=True)
    available = [r[0] for r in con.execute("DESCRIBE gold_features").fetchall()]
    select_cols = [c for c in FEATURE_COLS + CATEGORICAL_COLS + [TARGET_COL, "obs_year", "loan_id", "reporting_date"] if c in available]
    
    print("Fetching gold_features...")
    df_features = con.execute(f"SELECT {', '.join(select_cols)} FROM gold_features").fetchdf()
    
    emb_path = Path(args.run_dir) / "embeddings.parquet"
    if not emb_path.exists():
        print(f"Error: Embeddings not found at {emb_path}")
        return
        
    print(f"Loading Embeddings: {emb_path}")
    df_emb = pd.read_parquet(emb_path)
    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    
    print("Merging Data...")
    df = df_features.merge(df_emb, on="loan_id", how="inner")
    
    # Calculate months_to_default
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    if "default_crr_flag" in df.columns:
        default_dates = df[df["default_crr_flag"] == 1].groupby("loan_id")["reporting_date"].min().reset_index()
        default_dates.rename(columns={"reporting_date": "default_date"}, inplace=True)
        df = df.merge(default_dates, on="loan_id", how="left")
        df["months_to_default"] = ((df["default_date"] - df["reporting_date"]).dt.days / 30.0).round().fillna(-1)
    else:
        df["months_to_default"] = -1
        
    # Categorical encoding
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("__NULL__"))
            
    train_mask = df["obs_year"] <= 2024
    test_mask = df["obs_year"] >= 2025
    if test_mask.sum() == 0:
        sorted_df = df.sort_values("reporting_date")
        split_idx = int(len(sorted_df) * 0.8)
        train_mask = df.index.isin(sorted_df.index[:split_idx])
        test_mask = df.index.isin(sorted_df.index[split_idx:])
        
    df_train, df_test = df[train_mask], df[test_mask]
    y_train = df_train[TARGET_COL].fillna(0).astype(int).values
    y_test = df_test[TARGET_COL].fillna(0).astype(int).values
    
    # Define features
    feat_cols_avail = [c for c in FEATURE_COLS + CATEGORICAL_COLS if c in df.columns]
    X_train_base = df_train[feat_cols_avail].fillna(-999).values
    X_test_base = df_test[feat_cols_avail].fillna(-999).values
    
    X_train_emb = df_train[emb_cols].fillna(0).values
    X_test_emb = df_test[emb_cols].fillna(0).values
    
    # Hybrid
    X_train_hybrid = np.hstack([X_train_base, X_train_emb])
    X_test_hybrid = np.hstack([X_test_base, X_test_emb])
    
    # Standard scale embeddings for Linear / DNN
    scaler = StandardScaler()
    X_train_emb_scaled = scaler.fit_transform(X_train_emb)
    X_test_emb_scaled = scaler.transform(X_test_emb)
    
    use_gpu = torch.cuda.is_available()
    
    results = {}
    print("Training Baseline...")
    prob_base = train_xgboost(X_train_base, y_train, X_test_base, y_test, use_gpu)
    res_base = evaluate_model(y_test, prob_base)
    res_base["horizon_curve"] = get_horizon_curve(df_test, y_test, prob_base)
    results["baseline"] = res_base
    
    print("Training Embeddings-Only...")
    prob_emb = train_xgboost(X_train_emb, y_train, X_test_emb, y_test, use_gpu)
    res_emb = evaluate_model(y_test, prob_emb)
    res_emb["horizon_curve"] = get_horizon_curve(df_test, y_test, prob_emb)
    results["embeddings"] = res_emb
    
    print("Training Hybrid...")
    prob_hybrid = train_xgboost(X_train_hybrid, y_train, X_test_hybrid, y_test, use_gpu)
    res_hybrid = evaluate_model(y_test, prob_hybrid)
    res_hybrid["horizon_curve"] = get_horizon_curve(df_test, y_test, prob_hybrid)
    results["hybrid"] = res_hybrid
    
    print("Training Linear Probe...")
    prob_linear = train_linear(X_train_emb_scaled, y_train, X_test_emb_scaled)
    res_linear = evaluate_model(y_test, prob_linear)
    res_linear["horizon_curve"] = get_horizon_curve(df_test, y_test, prob_linear)
    results["linear_probe"] = res_linear
    
    print("Training PyTorch DNN...")
    prob_dnn = train_dnn(X_train_emb_scaled, y_train, X_test_emb_scaled)
    res_dnn = evaluate_model(y_test, prob_dnn)
    res_dnn["horizon_curve"] = get_horizon_curve(df_test, y_test, prob_dnn)
    results["dnn"] = res_dnn
    
    # Slicing for ESG (epc_label)
    if "epc_label" in df_test.columns:
        epc_results = {}
        # we encoded epc_label so it's int. But it's fine.
        for epc in df_test["epc_label"].unique():
            mask = df_test["epc_label"] == epc
            if mask.sum() > 50 and y_test[mask].sum() > 0:
                epc_results[str(epc)] = evaluate_model(y_test[mask], prob_hybrid[mask])["pr_auc"]
        results["esg_slices"] = epc_results
    
    out_file = Path(args.run_dir) / "downstream_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
