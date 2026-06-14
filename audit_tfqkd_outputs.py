
#!/usr/bin/env python3
"""
Audit TF-QKD datasets, runs, and analysis outputs.

Checks:
- CSV/Parquet integrity
- missing / NaN / inf / duplicate rows
- zero-heavy / constant numeric columns
- split leakage
- label balance
- sequence layout consistency
- run directory artifacts
- analysis PDF inventory
- paper-ready export inventory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TIMESTEP_RE = re.compile(r"^(?P<base>.+)_t(?P<t>\d+)$")
PDF_REQUIRED_HINTS = [
    "confusion",
    "supervised_metrics",
    "anomaly_metrics",
    "attention",
    "tsne",
    "drift_phase_lock_error_rad",
    "drift_qber_phase",
    "roc",
    "pr",
]


def safe_read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=True, na_filter=True)


def infer_feature_layout(columns: Sequence[str]) -> Tuple[List[str], int]:
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
        zero_frac = (numeric.fillna(0) == 0).mean().sort_values(ascending=False)
        out["top_zero_fraction_cols"] = {k: round(float(v), 4) for k, v in zero_frac.head(15).items() if float(v) >= 0.5}
        out["all_zero_numeric_columns"] = [c for c in numeric.columns if (numeric[c].fillna(0) == 0).all()][:20]
        low_var = [c for c in numeric.columns if len(numeric[c].dropna()) > 1 and float(numeric[c].var()) < 1e-12]
        if low_var:
            out["near_constant_numeric_columns"] = low_var[:20]

    if "label" in df.columns:
        counts = df["label"].astype(str).value_counts(dropna=False)
        out["label_counts"] = counts.to_dict()
        out["label_balance_ratio"] = float(counts.max() / max(counts.min(), 1))
    if "split" in df.columns:
        out["split_counts"] = df["split"].astype(str).value_counts(dropna=False).to_dict()
        if "sample_id" in df.columns:
            leak = df.groupby("sample_id")["split"].nunique()
            out["split_leakage_sample_ids"] = int((leak > 1).sum())

    seq_cols = [c for c in df.columns if TIMESTEP_RE.match(c)]
    bases, seq_len = infer_feature_layout(seq_cols)
    out["n_sequence_columns"] = len(seq_cols)
    out["n_bases"] = len(bases)
    out["sequence_len"] = int(seq_len)

    return out


def scan_run_dir(run_dir: Path) -> Dict[str, Any]:
    files = {p.name for p in run_dir.iterdir() if p.is_file()}
    expected = ["autoencoder.pt", "lstm.pt", "qlstm.pt", "transformer.pt", "metrics.csv", "scaler.pkl", "label_encoder.pkl", "feature_layout.json", "run_config.json"]
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "missing_files": [f for f in expected if f not in files],
        "present_files": sorted(files),
    }
    mpath = run_dir / "metrics.csv"
    if mpath.exists():
        metrics = safe_read(mpath)
        summary["metrics_rows"] = int(len(metrics))
        summary["metrics_cols"] = int(metrics.shape[1])
        if "model" in metrics.columns:
            summary["models"] = metrics["model"].astype(str).tolist()
        summary["tasks"] = metrics["task"].astype(str).value_counts().to_dict() if "task" in metrics.columns else {}
    else:
        summary["metrics_missing"] = True
    return summary


def scan_analysis_dir(analysis_dir: Path) -> Dict[str, Any]:
    files = list(analysis_dir.glob("*.pdf")) + list(analysis_dir.glob("*.csv")) + list(analysis_dir.glob("*.json")) + list(analysis_dir.glob("*.md"))
    pdfs = sorted([p.name for p in analysis_dir.glob("*.pdf")])
    csvs = sorted([p.name for p in analysis_dir.glob("*.csv")])
    summary: Dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "pdf_count": len(pdfs),
        "csv_count": len(csvs),
        "pdf_files": pdfs[:40],
        "csv_files": csvs[:40],
    }
    mpath = analysis_dir / "metrics.csv"
    if mpath.exists():
        metrics = safe_read(mpath)
        summary["metrics_rows"] = int(len(metrics))
        summary["metrics_cols"] = int(metrics.shape[1])
        if "task" in metrics.columns:
            summary["task_counts"] = metrics["task"].astype(str).value_counts().to_dict()
        if "model" in metrics.columns:
            summary["models"] = metrics["model"].astype(str).tolist()

    missing_hints = [hint for hint in PDF_REQUIRED_HINTS if not any(hint in p.lower() for p in pdfs)]
    if missing_hints:
        summary["missing_pdf_hints"] = missing_hints
    return summary


def scan_report_dir(report_dir: Path) -> Dict[str, Any]:
    summary = {"report_dir": str(report_dir), "exists": report_dir.exists()}
    if (report_dir / "audit_report.md").exists():
        summary["audit_report"] = str(report_dir / "audit_report.md")
    if (report_dir / "audit_report.json").exists():
        summary["audit_json"] = str(report_dir / "audit_report.json")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*", default=[], help="Dataset CSV/parquet files")
    p.add_argument("--runs", nargs="*", default=[], help="Run directories with checkpoints/metrics")
    p.add_argument("--reports", nargs="*", default=[], help="Analysis/report directories to inspect")
    p.add_argument("--outdir", type=str, default="audit_out")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"datasets": [], "runs": [], "reports": []}
    for ds in args.datasets:
        pth = Path(ds)
        if not pth.exists():
            results["datasets"].append({"name": str(pth), "missing": True})
            continue
        df = safe_read(pth)
        results["datasets"].append(summarize_dataframe(df, pth.name))
    for rd in args.runs:
        pth = Path(rd)
        if not pth.exists():
            results["runs"].append({"run_dir": str(pth), "missing": True})
            continue
        results["runs"].append(scan_run_dir(pth))
    for rep in args.reports:
        pth = Path(rep)
        if not pth.exists():
            results["reports"].append({"report_dir": str(pth), "missing": True})
            continue
        if (pth / "metrics.csv").exists() or any(f.suffix.lower() == ".pdf" for f in pth.iterdir()):
            results["reports"].append(scan_analysis_dir(pth))
        else:
            results["reports"].append(scan_report_dir(pth))

    md = ["# TF-QKD audit report", ""]
    md.append("## Datasets")
    for item in results["datasets"]:
        md.append(f"### {item.get('name')}")
        for k, v in item.items():
            if k == "name":
                continue
            md.append(f"- {k}: {v}")
        md.append("")
    md.append("## Run directories")
    for item in results["runs"]:
        md.append(f"### {item.get('run_dir')}")
        for k, v in item.items():
            if k == "run_dir":
                continue
            md.append(f"- {k}: {v}")
        md.append("")
    md.append("## Reports / analysis directories")
    for item in results["reports"]:
        md.append(f"### {item.get('analysis_dir', item.get('report_dir'))}")
        for k, v in item.items():
            if k in {"analysis_dir", "report_dir"}:
                continue
            md.append(f"- {k}: {v}")
        md.append("")

    (outdir / "audit_report.md").write_text("\n".join(md), encoding="utf-8")
    (outdir / "audit_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n".join(md))
    print(f"\nSaved: {outdir / 'audit_report.md'}")
    print(f"Saved: {outdir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
