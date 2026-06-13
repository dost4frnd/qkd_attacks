#!/usr/bin/env python3
"""
audit_tfqkd_outputs.py

Lightweight audit script for TF-QKD experiments.

Checks:
- dataset CSV / parquet integrity
- NaNs, infs, empty cells, zero-heavy columns
- duplicate rows
- flattened sequence structure
- train/val/test split leakage
- label distribution / class imbalance
- probability sanity (predictions CSVs)
- metrics sanity (missing / zero / out-of-range / suspicious values)
- artifact presence in run directories
- cross-domain summary file sanity

Examples:
    python audit_tfqkd_outputs.py \
      --datasets tfqkd_datasets/tfqkd_clean/tfqkd_flat.csv tfqkd_datasets/tfqkd_drift/tfqkd_flat.csv \
      --runs runs_qkd runs_clean_s runs_drift_s runs_asym_s runs_unknown_s \
      --reports analysis_clean analysis_drift analysis_asym analysis_unknown cross_domain

    python audit_tfqkd_outputs.py --datasets tfqkd_datasets/tfqkd_clean/tfqkd_flat.csv --runs runs_qkd

Outputs:
    audit_report.md
    audit_report.json
    optional per-file summary CSVs in --outdir
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

TIMESTEP_RE = re.compile(r"^(?P<base>.+)_t(?P<t>\d+)$")
PROB_COL_RE = re.compile(r"^p_(.+)$")


def safe_read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=True, na_filter=True)


def infer_feature_layout(columns: List[str]) -> Tuple[List[str], int]:
    groups: Dict[str, List[int]] = {}
    for c in columns:
        m = TIMESTEP_RE.match(c)
        if not m:
            continue
        groups.setdefault(m.group("base"), []).append(int(m.group("t")))
    if not groups:
        return [], 0
    bases = sorted(groups.keys())
    lens = {len(set(v)) for v in groups.values()}
    if len(lens) != 1:
        return bases, -1
    return bases, lens.pop()


def find_sequence_issues(df: pd.DataFrame) -> Dict[str, Any]:
    res = {}
    seq_cols = [c for c in df.columns if TIMESTEP_RE.match(c)]
    bases, seq_len = infer_feature_layout(seq_cols)
    res["n_sequence_columns"] = len(seq_cols)
    res["n_bases"] = len(bases)
    res["sequence_len"] = seq_len

    if seq_cols:
        # missing timestep columns per base
        missing = {}
        for base in bases:
            expected = [f"{base}_t{t:02d}" for t in range(seq_len)]
            absent = [c for c in expected if c not in df.columns]
            if absent:
                missing[base] = absent[:5]
        if missing:
            res["missing_sequence_columns"] = missing

        # detect constant / near-constant columns
        zero_heavy = {}
        const_cols = []
        for c in seq_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() == 0:
                const_cols.append(c)
                continue
            if s.nunique(dropna=True) <= 1:
                const_cols.append(c)
            frac_zero = float((s.fillna(0) == 0).mean())
            if frac_zero >= 0.98:
                zero_heavy[c] = round(frac_zero, 4)
        if const_cols:
            res["constant_sequence_columns"] = const_cols[:20]
        if zero_heavy:
            # keep only top 20
            res["zero_heavy_sequence_columns"] = dict(sorted(zero_heavy.items(), key=lambda kv: kv[1], reverse=True)[:20])
    return res


def summarize_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "nan_cells": int(df.isna().sum().sum()),
        "inf_cells": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()) if not df.select_dtypes(include=[np.number]).empty else 0,
    }

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        out["numeric_cols"] = int(numeric.shape[1])
        zero_frac = (numeric.fillna(0) == 0).mean().sort_values(ascending=False)
        out["top_zero_fraction_cols"] = {k: round(float(v), 4) for k, v in zero_frac.head(15).items() if float(v) >= 0.5}

        # suspicious all-zero columns
        all_zero_cols = [c for c in numeric.columns if (numeric[c].fillna(0) == 0).all()]
        if all_zero_cols:
            out["all_zero_numeric_columns"] = all_zero_cols[:20]

        # very low variance
        low_var = []
        for c in numeric.columns:
            s = numeric[c].dropna()
            if len(s) > 1 and float(s.var()) < 1e-12:
                low_var.append(c)
        if low_var:
            out["near_constant_numeric_columns"] = low_var[:20]

    if "label" in df.columns:
        counts = df["label"].astype(str).value_counts(dropna=False)
        out["label_counts"] = counts.to_dict()
        out["label_balance_ratio"] = float(counts.max() / max(counts.min(), 1))

    if "split" in df.columns:
        out["split_counts"] = df["split"].astype(str).value_counts(dropna=False).to_dict()
        if {"sample_id", "split"}.issubset(df.columns):
            # same sample_id in multiple splits is suspicious
            split_counts = df.groupby("sample_id")["split"].nunique()
            leaked = int((split_counts > 1).sum())
            if leaked:
                out["split_leakage_sample_ids"] = leaked

    # flattened seq structure
    out.update(find_sequence_issues(df))

    return out


def load_metrics(path: Path) -> Dict[str, Any]:
    df = safe_read(path)
    result: Dict[str, Any] = {"file": str(path), "rows": int(len(df)), "cols": int(df.shape[1])}

    if df.empty:
        result["issue"] = "empty_metrics_file"
        return result

    if "model" in df.columns:
        result["models"] = df["model"].astype(str).tolist()

    # generic numeric sanity checks
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if col.lower() in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "roc_auc_ovr_macro", "average_precision"}:
            bad = s[(s < -1e-9) | (s > 1 + 1e-9)]
            if len(bad):
                result.setdefault("out_of_range_metrics", {})[col] = int(len(bad))
            if s.fillna(0).eq(0).all():
                result.setdefault("all_zero_metrics", []).append(col)
        if col.lower() == "threshold" and (s <= 0).any():
            result.setdefault("nonpositive_thresholds", int((s <= 0).sum()))
    return result


def load_predictions(path: Path) -> Dict[str, Any]:
    df = safe_read(path)
    out: Dict[str, Any] = {"file": str(path), "rows": int(len(df)), "cols": int(df.shape[1])}
    if df.empty:
        out["issue"] = "empty_predictions_file"
        return out

    prob_cols = [c for c in df.columns if PROB_COL_RE.match(c)]
    if prob_cols:
        probs = df[prob_cols].apply(pd.to_numeric, errors="coerce")
        row_sums = probs.sum(axis=1, skipna=False)
        bad_sum = ((row_sums < 0.95) | (row_sums > 1.05)).sum()
        out["probability_columns"] = prob_cols
        out["bad_probability_row_sums"] = int(bad_sum)
        out["nan_probability_cells"] = int(probs.isna().sum().sum())
        out["prob_min"] = float(np.nanmin(probs.values))
        out["prob_max"] = float(np.nanmax(probs.values))
        if (probs < -1e-9).any().any() or (probs > 1 + 1e-9).any().any():
            out["prob_out_of_range"] = True

    if "recon_error" in df.columns:
        s = pd.to_numeric(df["recon_error"], errors="coerce")
        out["recon_error_min"] = float(s.min())
        out["recon_error_max"] = float(s.max())
        out["recon_error_nonpositive"] = int((s <= 0).sum())

    if "y_true" in df.columns and "y_pred" in df.columns:
        mismatch = int((df["y_true"].astype(str) != df["y_pred"].astype(str)).sum())
        out["label_mismatch_count"] = mismatch

    return out


def scan_run_dir(run_dir: Path) -> Dict[str, Any]:
    files = {p.name for p in run_dir.iterdir() if p.is_file()}
    expected = ["autoencoder.pt", "lstm.pt", "qlstm.pt", "transformer.pt", "metrics.csv", "scaler.pkl", "label_encoder.pkl", "feature_layout.json", "run_config.json"]
    missing = [f for f in expected if f not in files]

    out = {
        "run_dir": str(run_dir),
        "present_files": sorted(list(files)),
        "missing_files": missing,
        "checkpoints_present": [f for f in ["autoencoder.pt", "lstm.pt", "qlstm.pt", "transformer.pt"] if f in files],
    }

    mpath = run_dir / "metrics.csv"
    if mpath.exists():
        out["metrics"] = load_metrics(mpath)
    else:
        out["metrics"] = {"issue": "missing_metrics_csv"}

    # optional prediction tables
    pred_files = sorted([p for p in run_dir.glob("*predictions.csv")] + [p for p in run_dir.glob("*scores.csv")])
    pred_summaries = []
    for p in pred_files:
        try:
            pred_summaries.append(load_predictions(p))
        except Exception as e:
            pred_summaries.append({"file": str(p), "issue": f"failed_to_read: {e}"})
    out["predictions"] = pred_summaries
    return out


def scan_report_dir(report_dir: Path) -> Dict[str, Any]:
    files = {p.name for p in report_dir.iterdir() if p.is_file()}
    pdfs = sorted([p.name for p in report_dir.glob("*.pdf")])
    csvs = sorted([p.name for p in report_dir.glob("*.csv")])
    out = {
        "report_dir": str(report_dir),
        "pdf_count": len(pdfs),
        "csv_count": len(csvs),
        "pdf_files": pdfs[:50],
        "csv_files": csvs[:50],
    }
    if "metrics.csv" in files:
        out["metrics"] = load_metrics(report_dir / "metrics.csv")
    if out.get("pdf_count", 0) == 0:
        out["issue"] = "no_pdf_figures_found"
    return out


def build_text_report(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# TF-QKD audit report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- datasets checked: {len(results.get('datasets', []))}")
    lines.append(f"- run dirs checked: {len(results.get('runs', []))}")
    lines.append(f"- report dirs checked: {len(results.get('reports', []))}")
    lines.append("")

    if results.get("dataset_issues"):
        lines.append("## Dataset issues")
        for ds in results["dataset_issues"]:
            lines.append(f"### {ds['name']}")
            for k, v in ds.items():
                if k == "name":
                    continue
                lines.append(f"- {k}: {v}")
            lines.append("")

    if results.get("run_issues"):
        lines.append("## Run directory issues")
        for rd in results["run_issues"]:
            lines.append(f"### {rd['run_dir']}")
            if rd.get("missing_files"):
                lines.append(f"- missing_files: {rd['missing_files']}")
            metrics = rd.get("metrics", {})
            if metrics:
                for k, v in metrics.items():
                    if k == "file":
                        continue
                    lines.append(f"- metrics.{k}: {v}")
            for pred in rd.get("predictions", []):
                lines.append(f"- pred_file: {pred.get('file')}")
                for k, v in pred.items():
                    if k == "file":
                        continue
                    lines.append(f"  - {k}: {v}")
            lines.append("")

    if results.get("report_issues"):
        lines.append("## Analysis report issues")
        for rep in results["report_issues"]:
            lines.append(f"### {rep['report_dir']}")
            lines.append(f"- pdf_count: {rep.get('pdf_count', 0)}")
            lines.append(f"- csv_count: {rep.get('csv_count', 0)}")
            if rep.get("metrics"):
                lines.append(f"- metrics rows: {rep['metrics'].get('rows')}")
                lines.append(f"- metrics cols: {rep['metrics'].get('cols')}")
            lines.append("")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=[], help="Dataset CSV/parquet files to inspect")
    p.add_argument("--runs", nargs="*", default=[], help="Run directories to inspect")
    p.add_argument("--reports", nargs="*", default=[], help="Analysis report directories to inspect")
    p.add_argument("--outdir", type=str, default="audit_out")
    p.add_argument("--save-json", action="store_true", help="Save JSON summary")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"datasets": [], "runs": [], "reports": []}
    dataset_issues = []
    run_issues = []
    report_issues = []

    for d in args.datasets:
        path = Path(d)
        if not path.exists():
            dataset_issues.append({"name": str(path), "issue": "missing_file"})
            continue
        try:
            df = safe_read(path)
            summary = summarize_dataframe(df, path.name)
            results["datasets"].append(summary)
            dataset_issues.append(summary)
        except Exception as e:
            dataset_issues.append({"name": str(path), "issue": f"failed_to_read: {e}"})

    for r in args.runs:
        path = Path(r)
        if not path.exists():
            run_issues.append({"run_dir": str(path), "issue": "missing_dir"})
            continue
        try:
            summary = scan_run_dir(path)
            results["runs"].append(summary)
            run_issues.append(summary)
        except Exception as e:
            run_issues.append({"run_dir": str(path), "issue": f"failed_to_scan: {e}"})

    for rep in args.reports:
        path = Path(rep)
        if not path.exists():
            report_issues.append({"report_dir": str(path), "issue": "missing_dir"})
            continue
        try:
            summary = scan_report_dir(path)
            results["reports"].append(summary)
            report_issues.append(summary)
        except Exception as e:
            report_issues.append({"report_dir": str(path), "issue": f"failed_to_scan: {e}"})

    results["dataset_issues"] = dataset_issues
    results["run_issues"] = run_issues
    results["report_issues"] = report_issues

    md = build_text_report(results)
    (outdir / "audit_report.md").write_text(md, encoding="utf-8")

    if args.save_json or True:
        (outdir / "audit_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Console summary
    print(md)
    print(f"\nSaved report to: {outdir/'audit_report.md'}")
    print(f"Saved JSON to: {outdir/'audit_report.json'}")


if __name__ == "__main__":
    main()
