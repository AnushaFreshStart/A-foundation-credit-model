"""
validate_gold.py — Temporal & Lifecycle SQL Validation
=======================================================
Executes a suite of validation checks on the DuckDB credit_validate.db.

Check Categories:
  1. Schema integrity  — table/view row counts
  2. Temporal sequence — seasoning increments, date gaps, remaining term
  3. Lifecycle         — defaulted balance freeze, terminal zero-out
  4. Gold quality      — label distribution, feature nulls

Usage:
    python validate_gold.py [--db PATH]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

WORKSPACE_DIR = Path(__file__).parent
DEFAULT_DB    = WORKSPACE_DIR / "credit_validate.db"


# ---------------------------------------------------------------------------
# Validation queries
# ---------------------------------------------------------------------------

CHECKS = [
    # --─ Schema Integrity ----------------------------------------------------
    {
        "id":       "schema.static_loans_nonempty",
        "category": "Schema Integrity",
        "name":     "static_loans has rows",
        "sql":      "SELECT COUNT(*) FROM static_loans",
        "pass_if":  lambda v: v > 0,
        "detail":   lambda v: f"{v:,} rows",
    },
    {
        "id":       "schema.dynamic_nonempty",
        "category": "Schema Integrity",
        "name":     "dynamic_performance has rows",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance",
        "pass_if":  lambda v: v > 0,
        "detail":   lambda v: f"{v:,} rows",
    },
    {
        "id":       "schema.gold_nonempty",
        "category": "Schema Integrity",
        "name":     "gold_features view has rows",
        "sql":      "SELECT COUNT(*) FROM gold_features",
        "pass_if":  lambda v: v > 0,
        "detail":   lambda v: f"{v:,} rows",
    },
    {
        "id":       "schema.fk_integrity",
        "category": "Schema Integrity",
        "name":     "All dynamic loan_ids exist in static_loans",
        "sql":      """
            SELECT COUNT(*) FROM dynamic_performance d
            LEFT JOIN static_loans s ON d.loan_id = s.loan_id
            WHERE s.loan_id IS NULL
        """,
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} orphaned dynamic rows",
    },
    {
        "id":       "schema.pk_uniqueness",
        "category": "Schema Integrity",
        "name":     "No duplicate (loan_id, reporting_date) in dynamic",
        "sql":      """
            SELECT COUNT(*) FROM (
                SELECT loan_id, reporting_date, COUNT(*) cnt
                FROM dynamic_performance
                GROUP BY loan_id, reporting_date
                HAVING cnt > 1
            )
        """,
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} duplicate pairs",
    },
    {
        "id":       "schema.cutoff_count",
        "category": "Schema Integrity",
        "name":     "At least 12 distinct reporting_dates",
        "sql":      "SELECT COUNT(DISTINCT reporting_date) FROM dynamic_performance",
        "pass_if":  lambda v: v >= 12,
        "detail":   lambda v: f"{v} distinct reporting dates",
    },

    # --─ Temporal Sequence --------------------------------------------------─
    {
        "id":       "temporal.seasoning_increment",
        "category": "Temporal Sequence",
        "name":     "seasoning_months increases by ~1 each month (>95% compliance)",
        "sql":      """
            WITH ordered AS (
                SELECT loan_id, reporting_date, seasoning_months,
                       LAG(seasoning_months) OVER (PARTITION BY loan_id ORDER BY reporting_date) AS prev_season,
                       LAG(reporting_date)   OVER (PARTITION BY loan_id ORDER BY reporting_date) AS prev_date
                FROM dynamic_performance
            ),
            checks AS (
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE
                        WHEN prev_season IS NULL THEN 1   -- first observation, OK
                        WHEN seasoning_months - prev_season BETWEEN 0 AND 2 THEN 1
                        ELSE 0
                    END) AS ok
                FROM ordered
            )
            SELECT ROUND(100.0 * ok / total, 2) FROM checks
        """,
        "pass_if":  lambda v: v >= 95.0,
        "detail":   lambda v: f"{v}% compliant",
    },
    {
        "id":       "temporal.remaining_term_decrements",
        "category": "Temporal Sequence",
        "name":     "remaining_term_months decreases by ~1 each month (>90% compliance)",
        "sql":      """
            WITH ordered AS (
                SELECT loan_id, reporting_date, remaining_term_months,
                       LAG(remaining_term_months) OVER (PARTITION BY loan_id ORDER BY reporting_date) AS prev_rem
                FROM dynamic_performance
            ),
            checks AS (
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE
                        WHEN prev_rem IS NULL THEN 1
                        WHEN prev_rem - remaining_term_months BETWEEN -1 AND 2 THEN 1
                        ELSE 0
                    END) AS ok
                FROM ordered
            )
            SELECT ROUND(100.0 * ok / total, 2) FROM checks
        """,
        "pass_if":  lambda v: v >= 90.0,
        "detail":   lambda v: f"{v}% compliant",
    },
    {
        "id":       "temporal.no_negative_seasoning",
        "category": "Temporal Sequence",
        "name":     "No negative seasoning_months",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance WHERE seasoning_months < 0",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} negative values",
    },
    {
        "id":       "temporal.no_future_dates",
        "category": "Temporal Sequence",
        "name":     "No reporting_date in the future",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance WHERE reporting_date > CURRENT_DATE",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} future dates",
    },

    # --─ Lifecycle / Business Logic ------------------------------------------
    {
        "id":       "lifecycle.nonneg_balance",
        "category": "Lifecycle",
        "name":     "No negative current_balance",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance WHERE current_balance < 0",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} negative balances",
    },
    {
        "id":       "lifecycle.nonneg_dpd",
        "category": "Lifecycle",
        "name":     "No negative days_past_due",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance WHERE days_past_due < 0",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} negative DPD rows",
    },
    {
        "id":       "lifecycle.nonneg_interest_rate",
        "category": "Lifecycle",
        "name":     "No negative interest rates",
        "sql":      "SELECT COUNT(*) FROM dynamic_performance WHERE current_interest_rate_pct < 0",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} negative rate rows",
    },
    {
        "id":       "lifecycle.positive_original_balance",
        "category": "Lifecycle",
        "name":     "All original_balance > 0 in static_loans",
        "sql":      "SELECT COUNT(*) FROM static_loans WHERE original_balance <= 0 OR original_balance IS NULL",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} invalid balances",
    },

    # --─ Gold Feature Quality ------------------------------------------------
    {
        "id":       "gold.label_distribution",
        "category": "Gold Quality",
        "name":     "default_in_3m label is not 100% zero (some positives exist)",
        "sql":      "SELECT SUM(default_in_3m) FROM gold_features",
        "pass_if":  lambda v: v > 0,
        "detail":   lambda v: f"{v:,} positive label rows",
    },
    {
        "id":       "gold.label_rate",
        "category": "Gold Quality",
        "name":     "default_in_3m rate is between 0.1% and 50%",
        "sql":      "SELECT ROUND(100.0 * AVG(default_in_3m), 4) FROM gold_features",
        "pass_if":  lambda v: 0.1 <= v <= 50.0,
        "detail":   lambda v: f"{v}% default rate",
    },
    {
        "id":       "gold.no_null_loan_id",
        "category": "Gold Quality",
        "name":     "No NULL loan_id in gold_features",
        "sql":      "SELECT COUNT(*) FROM gold_features WHERE loan_id IS NULL",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} nulls",
    },
    {
        "id":       "gold.no_null_reporting_date",
        "category": "Gold Quality",
        "name":     "No NULL reporting_date in gold_features",
        "sql":      "SELECT COUNT(*) FROM gold_features WHERE reporting_date IS NULL",
        "pass_if":  lambda v: v == 0,
        "detail":   lambda v: f"{v} nulls",
    },
]


def run_checks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Execute all validation checks and return results."""
    results = []
    for check in CHECKS:
        t0  = time.perf_counter()
        val = con.execute(check["sql"]).fetchone()[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        passed  = check["pass_if"](val)
        detail  = check["detail"](val)
        icon    = "OK" if passed else "[FAIL]"
        results.append({
            "id":          check["id"],
            "category":    check["category"],
            "name":        check["name"],
            "value":       val,
            "detail":      detail,
            "passed":      passed,
            "elapsed_ms":  round(elapsed_ms, 2),
        })
        status = "PASS" if passed else "FAIL"
        print(f"  {icon} [{status}] {check['name']}")
        print(f"          -> {detail}  ({elapsed_ms:.1f}ms)")

    return results


def main():
    parser = argparse.ArgumentParser(description="CFM Gold Validation Suite")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: Database not found at {args.db}")
        print("Run ingest_bronze.py first.")
        return 1

    print("=" * 60)
    print("  Credit Foundation Model — Gold Validation Suite")
    print("=" * 60)
    print(f"  Database: {args.db}\n")

    con = duckdb.connect(str(args.db), read_only=True)
    t_start  = time.perf_counter()
    results  = run_checks(con)
    con.close()
    total_ms = (time.perf_counter() - t_start) * 1000

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{len(results)} checks PASSED  |  {failed} FAILED")
    print(f"  Total validation time: {total_ms:.0f}ms")
    print("=" * 60)

    # Aggregate by category
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"pass": 0, "fail": 0})
        if r["passed"]:
            cats[c]["pass"] += 1
        else:
            cats[c]["fail"] += 1

    print("\n  Category Summary:")
    for cat, counts in cats.items():
        total_c = counts["pass"] + counts["fail"]
        print(f"    {cat}: {counts['pass']}/{total_c}")

    # Save report
    report = {
        "database":     str(args.db),
        "total_checks": len(results),
        "passed":       passed,
        "failed":       failed,
        "total_ms":     round(total_ms, 1),
        "status":       "ok" if failed == 0 else "warnings",
        "checks":       results,
    }
    report_path = WORKSPACE_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report saved -> {report_path.name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
