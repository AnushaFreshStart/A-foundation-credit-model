#!/usr/bin/env python
"""
train_foundation.py — CLI Entry Point for Credit Foundation Model Training
============================================================================
Called via subprocess from app.py or directly from command line.

Usage:
    python train_foundation.py --arch hybrid --strategy full --profile default
    python train_foundation.py --arch all --strategy pretrain_finetune
    python train_foundation.py --list-runs
    python train_foundation.py --compare-all
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import io
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
from config import TrainingConfig, VALID_ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(
        description="Credit Foundation Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model config
    parser.add_argument(
        "--arch", default="hybrid",
        choices=list(VALID_ARCHITECTURES) + ["all"],
        help="Model architecture (default: hybrid)",
    )
    parser.add_argument(
        "--strategy", default="full",
        choices=["full", "pretrain_only", "pretrain_finetune", "finetune_only", "joint_finetune"],
        help="Training strategy (default: full)",
    )
    parser.add_argument(
        "--profile", default="default",
        choices=["default", "small", "large", "fast", "custom"],
        help="Hyperparameter profile (default: default)",
    )

    # Overrides
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)

    # Paths
    parser.add_argument("--db", type=str, default=None, help="Path to credit_validate.db")
    parser.add_argument("--sequences", type=str, default=None, help="Path to sequences.parquet")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    # Commands
    parser.add_argument("--list-runs", action="store_true", help="List all training runs")
    parser.add_argument("--compare-all", action="store_true", help="Compare all runs")
    parser.add_argument("--compare", nargs="+", default=None, help="Compare specific run IDs")

    args = parser.parse_args()

    # Import heavy modules only after parsing args
    from trainer import CreditModelTrainer
    from run_manager import RunManager

    base_dir = Path(__file__).parent
    run_mgr = RunManager(base_dir)

    # --- Handle list/compare commands ---
    if args.list_runs:
        runs = run_mgr.list_runs()
        if not runs:
            print("No completed runs found.")
        else:
            print(f"\n{'─' * 80}")
            print(f"  {'Run ID':<45} {'Arch':<12} {'AUC':>6}  {'Gini':>6}  {'Time':>6}")
            print(f"{'─' * 80}")
            for r in runs:
                print(
                    f"  {r['run_id']:<45} "
                    f"{r['architecture']:<12} "
                    f"{r['auc_roc_default']:>6.4f}  "
                    f"{r['gini_default']:>6.4f}  "
                    f"{r['total_time_s']:>5.0f}s"
                )
            print(f"{'─' * 80}")
        print(json.dumps(runs, indent=2))
        return

    if args.compare_all or args.compare:
        comparison = run_mgr.compare_runs(args.compare)
        print(json.dumps(comparison, indent=2))
        return

    # --- Training ---
    architectures = list(VALID_ARCHITECTURES) if args.arch == "all" else [args.arch]

    all_results = []

    for arch in architectures:
        print(f"\n{'═' * 60}")
        print(f"  TRAINING: {arch.upper()} / {args.strategy} / {args.profile}")
        print(f"{'═' * 60}")

        # Build config
        overrides = {"architecture": arch, "strategy": args.strategy}
        if args.embed_dim is not None:
            overrides["embed_dim"] = args.embed_dim
        if args.n_heads is not None:
            overrides["n_heads"] = args.n_heads
        if args.n_layers is not None:
            overrides["n_layers"] = args.n_layers
        if args.lr is not None:
            overrides["learning_rate"] = args.lr
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.pretrain_epochs is not None:
            overrides["pretrain_epochs"] = args.pretrain_epochs
        if args.joint_epochs is not None:
            overrides["joint_epochs"] = args.joint_epochs
        if args.finetune_epochs is not None:
            overrides["finetune_epochs"] = args.finetune_epochs
        if args.db:
            overrides["db_path"] = args.db
        if args.sequences:
            overrides["sequences_path"] = args.sequences
        if args.output:
            overrides["output_dir"] = args.output

        config = TrainingConfig.load_profile(args.profile, **overrides)
        config.resolve_paths()

        warnings = config.validate()
        for w in warnings:
            print(f"  ⚠ {w}")

        print(config.summary())

        # Start run
        run_id = run_mgr.start_run(config)

        # Train
        trainer = CreditModelTrainer(config)
        results = trainer.train()
        results["run_id"] = run_id

        # Save
        run_mgr.save_run(run_id, results, checkpoint=trainer.get_checkpoint())

        all_results.append({
            "run_id": run_id,
            "architecture": arch,
            "results": results,
        })

    # Print final summary
    if len(architectures) > 1:
        print(f"\n{'═' * 60}")
        print(f"  COMPARISON ACROSS {len(architectures)} ARCHITECTURES")
        print(f"{'═' * 60}")
        comparison = run_mgr.compare_runs([r["run_id"] for r in all_results])
        print(json.dumps(comparison, indent=2))

    # Output JSON for FastAPI capture
    print("\n---JSON_RESULTS_START---")
    print(json.dumps(all_results, indent=2, default=str))
    print("---JSON_RESULTS_END---")


if __name__ == "__main__":
    main()
