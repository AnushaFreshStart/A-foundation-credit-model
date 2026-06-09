"""
app.py — FastAPI Backend for Credit Foundation Model Dashboard
==============================================================
Provides REST endpoints to trigger pipeline stages and retrieve results.

Endpoints:
  GET  /                      -> Serve dashboard HTML
  GET  /api/status            -> DB row counts, file size, schema info
  POST /api/run/ingest        -> Run ingest_bronze.py
  POST /api/run/validate      -> Run validate_gold.py
  POST /api/run/train         -> Run baseline_model.py
  GET  /api/results/ingest    -> Return ingest_report.json
  GET  /api/results/validate  -> Return validation_report.json
  GET  /api/results/model     -> Return model_results.json

Usage:
    python app.py
    # Dashboard at http://localhost:8000
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
WORKSPACE    = Path(__file__).parent
PIPELINES_DIR = WORKSPACE / "pipelines"
DB_PATH      = PIPELINES_DIR / "credit_validate.db"
PYTHON       = sys.executable  # use same venv
# PYTHON = r"C:/venv/Scripts/python.exe"

app = FastAPI(title="Credit Foundation Model", version="1.0.0")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(WORKSPACE / "ui")), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_script(script_name: str, extra_args: list[str] = None) -> dict:
    """Run a pipeline script in a subprocess and return output."""
    if script_name == "build_sequences.py":
        script_path = WORKSPACE / "tokenize-sequences-app" / script_name
        cwd = WORKSPACE / "tokenize-sequences-app"
    else:
        script_path = PIPELINES_DIR / script_name
        cwd = PIPELINES_DIR

    cmd = [PYTHON, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    t0     = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
    )
    elapsed = time.perf_counter() - t0

    return {
        "returncode": result.returncode,
        "stdout":     result.stdout,
        "stderr":     result.stderr,
        "elapsed_s":  round(elapsed, 2),
        "success":    result.returncode == 0,
    }


def read_json_report(path: Path) -> dict:
    """Read a JSON report file if it exists."""
    if not path.exists():
        return {"error": f"{path.name} not found. Run the corresponding pipeline step first."}
    return json.loads(path.read_text(encoding="utf-8"))


def db_status() -> dict:
    """Query the DuckDB file for live row counts and metadata."""
    if not DB_PATH.exists():
        return {"db_exists": False, "db_path": str(DB_PATH)}

    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)

        counts = {}
        for tbl in ["static_loans", "dynamic_performance"]:
            try:
                counts[tbl] = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except Exception:
                counts[tbl] = None

        try:
            counts["gold_features"] = con.execute(
                "SELECT COUNT(*) FROM gold_features"
            ).fetchone()[0]
        except Exception:
            counts["gold_features"] = None

        # Date range
        try:
            dates = con.execute(
                "SELECT MIN(reporting_date), MAX(reporting_date), COUNT(DISTINCT reporting_date) FROM dynamic_performance"
            ).fetchone()
            date_range = {
                "min": str(dates[0]),
                "max": str(dates[1]),
                "cutoffs": dates[2],
            }
        except Exception:
            date_range = {}

        # Default rate
        try:
            default_rate = con.execute(
                "SELECT ROUND(100.0 * AVG(default_in_3m), 3) FROM gold_features"
            ).fetchone()[0]
        except Exception:
            default_rate = None

        con.close()

        db_size_mb = round(DB_PATH.stat().st_size / 1_048_576, 2)

        return {
            "db_exists":    True,
            "db_path":      str(DB_PATH),
            "db_size_mb":   db_size_mb,
            "row_counts":   counts,
            "date_range":   date_range,
            "default_rate": default_rate,
        }
    except Exception as e:
        return {"db_exists": True, "error": str(e)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    html_path = WORKSPACE / "ui" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """Return current DB stats and pipeline report existence."""
    status = db_status()
    status["reports"] = {
        "ingest_report":     (PIPELINES_DIR / "ingest_report.json").exists(),
        "validation_report": (PIPELINES_DIR / "validation_report.json").exists(),
        "model_results":     (PIPELINES_DIR / "model_results.json").exists(),
        "sequence_stats":    (WORKSPACE / "tokenize-sequences-result" / "sequence_stats.json").exists(),
        "tokenizer":         (WORKSPACE / "tokenize-sequences-result" / "tokenizer.json").exists(),
    }
    return JSONResponse(content=status)


@app.post("/api/run/ingest")
async def run_ingest():
    """Trigger the Bronze/Silver ingestion pipeline."""
    result = run_script("ingest_bronze.py")
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed:\n{result['stderr']}"
        )
    report = read_json_report(PIPELINES_DIR / "ingest_report.json")
    return JSONResponse(content={"run": result, "report": report})


@app.post("/api/run/validate")
async def run_validate():
    """Trigger the Gold validation suite."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=400, detail="Database not found. Run ingestion first.")
    result = run_script("validate_gold.py")
    report = read_json_report(PIPELINES_DIR / "validation_report.json")
    return JSONResponse(content={"run": result, "report": report})


@app.post("/api/run/train")
async def run_train():
    """Trigger XGBoost baseline training."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=400, detail="Database not found. Run ingestion first.")
    result = run_script("baseline_model.py")
    report = read_json_report(PIPELINES_DIR / "model_results.json")
    return JSONResponse(content={"run": result, "report": report})


@app.post("/api/run/tokenize")
async def run_tokenize():
    """Trigger Step 2: Tokenize and construct loan event sequences."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=400, detail="Database not found. Run ingestion first.")
    result = run_script("build_sequences.py")
    stats  = read_json_report(WORKSPACE / "tokenize-sequences-result" / "sequence_stats.json")
    return JSONResponse(content={"run": result, "report": stats})


@app.get("/api/results/sequences")
async def get_sequence_results():
    return JSONResponse(content=read_json_report(WORKSPACE / "tokenize-sequences-result" / "sequence_stats.json"))


@app.get("/api/results/tokenizer")
async def get_tokenizer_summary():
    return JSONResponse(content=read_json_report(WORKSPACE / "tokenize-sequences-result" / "tokenizer_summary.json"))


@app.get("/api/results/ingest")
async def get_ingest_results():
    return JSONResponse(content=read_json_report(PIPELINES_DIR / "ingest_report.json"))


@app.get("/api/results/validate")
async def get_validate_results():
    return JSONResponse(content=read_json_report(PIPELINES_DIR / "validation_report.json"))


@app.get("/api/results/model")
async def get_model_results():
    return JSONResponse(content=read_json_report(PIPELINES_DIR / "model_results.json"))


# ---------------------------------------------------------------------------
# Step 3: Foundation Model Training Endpoints
# ---------------------------------------------------------------------------

TRAIN_FM_DIR = WORKSPACE / "train-foundation-model"
TRAIN_FM_SCRIPT = TRAIN_FM_DIR / "train_foundation.py"


@app.post("/api/run/foundation")
async def run_foundation_training(
    arch: str = "hybrid",
    strategy: str = "full",
    profile: str = "default",
    pretrain_epochs: int | None = None,
    joint_epochs: int | None = None,
    finetune_epochs: int | None = None,
):
    """Trigger foundation model training with specified config."""
    seq_path = WORKSPACE / "tokenize-sequences-result" / "sequences.parquet"
    if not seq_path.exists():
        raise HTTPException(status_code=400, detail="Sequences not found. Run tokenization first.")

    args = [
        "--arch", arch,
        "--strategy", strategy,
        "--profile", profile,
        "--db", str(DB_PATH),
        "--sequences", str(seq_path),
    ]
    if pretrain_epochs is not None:
        args.extend(["--pretrain-epochs", str(pretrain_epochs)])
    if joint_epochs is not None:
        args.extend(["--joint-epochs", str(joint_epochs)])
    if finetune_epochs is not None:
        args.extend(["--finetune-epochs", str(finetune_epochs)])

    result = run_script_custom(TRAIN_FM_SCRIPT, args, cwd=TRAIN_FM_DIR)

    # Parse JSON results from stdout
    parsed = _parse_json_output(result["stdout"])

    return JSONResponse(content={
        "run": result,
        "results": parsed,
    })


@app.get("/api/foundation/runs")
async def list_foundation_runs():
    """List all foundation model training runs."""
    runs_dir = TRAIN_FM_DIR / "runs"
    if not runs_dir.exists():
        return JSONResponse(content={"runs": []})

    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "training_results.json"
        config_path = run_dir / "config.json"
        if not results_path.exists():
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

            ft = results.get("stages", {}).get("finetune", {}).get("metrics", {})
            default_m = ft.get("default", {}) if isinstance(ft, dict) else {}

            runs.append({
                "run_id": run_dir.name,
                "architecture": config.get("architecture", "?"),
                "strategy": config.get("strategy", "?"),
                "profile": config.get("profile", "?"),
                "auc_roc_default": default_m.get("auc_roc", 0),
                "gini_default": default_m.get("gini", 0),
                "total_params": results.get("total_params", 0),
                "total_time_s": results.get("total_time_s", 0),
                "has_checkpoint": (run_dir / "checkpoint.pt").exists(),
                "has_embeddings": (run_dir / "embeddings.parquet").exists(),
            })
        except Exception:
            continue

    runs.sort(key=lambda x: x.get("auc_roc_default", 0), reverse=True)
    return JSONResponse(content={"runs": runs})


@app.get("/api/foundation/run/{run_id}")
async def get_foundation_run(run_id: str):
    """Get detailed results for a specific training run."""
    results_path = TRAIN_FM_DIR / "runs" / run_id / "training_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return JSONResponse(content=json.loads(results_path.read_text(encoding="utf-8")))


@app.get("/api/foundation/compare")
async def compare_foundation_runs():
    """Compare all foundation model runs side by side."""
    runs_dir = TRAIN_FM_DIR / "runs"
    if not runs_dir.exists():
        return JSONResponse(content={"runs": [], "baseline": None})

    run_results = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_path = run_dir / "training_results.json"
        if not results_path.exists():
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            ft = results.get("stages", {}).get("finetune", {}).get("metrics", {})
            default_m = ft.get("default", {}) if isinstance(ft, dict) else {}

            run_results.append({
                "run_id": run_dir.name,
                "architecture": results.get("config", {}).get("architecture", "?"),
                "strategy": results.get("config", {}).get("strategy", "?"),
                "auc_roc_default": default_m.get("auc_roc", 0),
                "gini_default": default_m.get("gini", 0),
                "avg_precision": default_m.get("avg_precision", 0),
                "ks_statistic": default_m.get("ks_statistic", 0),
                "brier_score": default_m.get("brier_score", 0),
                "params": results.get("total_params", 0),
                "total_time_s": results.get("total_time_s", 0),
            })
        except Exception:
            continue

    run_results.sort(key=lambda x: x["auc_roc_default"], reverse=True)

    # Load XGBoost baseline
    baseline = None
    baseline_path = PIPELINES_DIR / "model_results.json"
    if baseline_path.exists():
        try:
            bl = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline = {
                "auc_roc": bl.get("auc_roc_test", 0),
                "gini": bl.get("gini_test", 0),
            }
        except Exception:
            pass

    return JSONResponse(content={"runs": run_results, "baseline": baseline})


def run_script_custom(script_path: Path, extra_args: list[str], cwd: Path) -> dict:
    """Run a script in subprocess from custom directory."""
    cmd = [PYTHON, str(script_path)] + extra_args
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(cwd))
    elapsed = time.perf_counter() - t0
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_s": round(elapsed, 2),
        "success": result.returncode == 0,
    }


def _parse_json_output(stdout: str) -> list | None:
    """Extract JSON results from CLI stdout (between markers)."""
    try:
        start = stdout.index("---JSON_RESULTS_START---")
        end = stdout.index("---JSON_RESULTS_END---")
        json_str = stdout[start + len("---JSON_RESULTS_START---"):end].strip()
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Step 4: Downstream Task Endpoints
# ---------------------------------------------------------------------------
DOWNSTREAM_SCRIPT = WORKSPACE / "embeddings-to-downstream" / "downstream_eval.py"

@app.post("/api/run/downstream/{run_id}")
async def run_downstream(run_id: str):
    """Trigger Downstream Evaluation for a specific Foundation Model run."""
    run_dir = TRAIN_FM_DIR / "runs" / run_id
    emb_path = run_dir / "embeddings.parquet"
    
    if not emb_path.exists():
        raise HTTPException(status_code=400, detail=f"Embeddings parquet not found for run {run_id}. Run Stage 3 first.")
        
    args = [
        "--run-id", run_id,
        "--db", str(DB_PATH),
        "--run-dir", str(run_dir),
    ]
    
    result = run_script_custom(DOWNSTREAM_SCRIPT, args, cwd=WORKSPACE / "embeddings-to-downstream")
    
    # Check if results file was written
    results_file = run_dir / "downstream_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=500, detail=f"Downstream eval failed to write results:\n{result['stderr']}")
        
    return JSONResponse(content={
        "success": True,
        "stdout": result["stdout"],
        "results": json.loads(results_file.read_text(encoding="utf-8"))
    })

@app.get("/api/downstream/results/{run_id}")
async def get_downstream_results(run_id: str):
    """Retrieve saved downstream evaluation results for a specific run."""
    results_file = TRAIN_FM_DIR / "runs" / run_id / "downstream_results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail=f"Downstream results not found for run '{run_id}'")
    return JSONResponse(content=json.loads(results_file.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Credit Foundation Model Dashboard")
    print("  http://localhost:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
