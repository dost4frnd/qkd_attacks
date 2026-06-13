#!/usr/bin/env bash
# Copy only paper-ready figures/tables into one flat directory.
# Converts raster plots to PDF for LaTeX embedding.

set -euo pipefail

OUTDIR="paper_ready_results"
KEEP_TOP=20
SOURCE_DIRS=()

usage() {
  cat <<EOF
Usage:
  bash paper_ready_export.sh [--outdir DIR] [--source-dirs DIR1 DIR2 ...] [--keep-top N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --keep-top)
      KEEP_TOP="$2"
      shift 2
      ;;
    --source-dirs)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SOURCE_DIRS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#SOURCE_DIRS[@]} -eq 0 ]]; then
  SOURCE_DIRS=(".")
fi

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*

python - "$OUTDIR" "$KEEP_TOP" "${SOURCE_DIRS[@]}" <<'PY'
from pathlib import Path
import fnmatch
import shutil
import sys
import re
from PIL import Image

outdir = Path(sys.argv[1])
keep_top = int(sys.argv[2])
source_dirs = [Path(p) for p in sys.argv[3:]]

outdir.mkdir(parents=True, exist_ok=True)
manifest = outdir / "paper_ready_manifest.md"
manifest.write_text("# Paper-ready export manifest\n\n", encoding="utf-8")

priority_patterns = [
    "cross_domain_accuracy",
    "cross_domain_f1_macro",
    "cross_domain_roc_auc",
    "qlstm_confusion_matrix",
    "lstm_confusion_matrix",
    "transformer_confusion_matrix",
    "metrics_comparison",
    "drift_phase_lock_error_rad",
    "drift_qber_phase",
    "tsne_transformer_embeddings",
    "transformer_attention",
    "autoencoder_roc",
    "autoencoder_pr",
    "autoencoder_scores_hist",
    "drift_visibility",
    "drift_ref_power_t_dbm",
    "drift_ref_wavelength_t_nm",
    "drift_coincidence_rate",
    "cross_domain_precision_macro",
    "cross_domain_recall_macro",
]

table_candidates = [
    "combined_metrics.csv",
    "metrics.csv",
    "audit_report.md",
    "audit_report.json",
    "anomaly_scores_by_class.csv",
]

def norm_stem(stem: str) -> str:
    stem = re.sub(r"^FIGURE_\d+_", "", stem)
    stem = re.sub(r"^FIG_\d+_", "", stem)
    return stem

def convert_to_pdf(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext == ".pdf":
        shutil.copy2(src, dst)
    elif ext in {".png", ".jpg", ".jpeg"}:
        img = Image.open(src).convert("RGB")
        img.save(dst, "PDF", resolution=300.0)
    elif ext == ".svg":
        # raster fallback via pillow if supported; if not, copy as-is with pdf suffix is not safe.
        # Try to let PIL open it; otherwise skip.
        img = Image.open(src).convert("RGB")
        img.save(dst, "PDF", resolution=300.0)
    else:
        raise ValueError(f"Unsupported figure type: {src}")

selected = 0
idx = 1
seen = set()

# choose candidates by priority over all source dirs
for pattern in priority_patterns:
    found = None
    for d in source_dirs:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            stem_norm = norm_stem(f.stem).lower()
            name_norm = f.name.lower()
            # exact match against the normalized stem prevents qlstm from matching lstm
            if stem_norm == pattern.lower() or name_norm == f"{pattern.lower()}.pdf" or name_norm == f"{pattern.lower()}.png":
                found = f
                break
        if found:
            break
    if not found:
        continue

    if found in seen:
        continue
    seen.add(found)

    # derive output name
    src_label = found.parent.name
    # Avoid generic container prefixes like figures/, results/, data/.
    if src_label in {"", ".", "figures", "results", "data", "paper_ready_results"}:
        src_label = ""
    stem = norm_stem(found.stem)
    if src_label:
        out_name = f"FIG_{idx:02d}_{src_label}_{stem}.pdf"
    else:
        out_name = f"FIG_{idx:02d}_{stem}.pdf"

    convert_to_pdf(found, outdir / out_name)
    manifest.write_text(manifest.read_text(encoding="utf-8") + f"- {out_name} <= {found}\n", encoding="utf-8")
    idx += 1
    selected += 1
    if selected >= keep_top:
        break

# tables / csv / markdown docs
for tbl in table_candidates:
    copied = False
    for d in source_dirs:
        if not d.exists():
            continue
        for f in [p for p in d.rglob(tbl) if p.is_file()]:
            out_name = f"{d.name}_{tbl}" if d.name not in {"", "."} else tbl
            shutil.copy2(f, outdir / out_name)
            manifest.write_text(manifest.read_text(encoding="utf-8") + f"- {out_name} <= {f}\n", encoding="utf-8")
            copied = True
            break
        if copied:
            break
    if not copied:
        manifest.write_text(manifest.read_text(encoding="utf-8") + f"- missing table: {tbl}\n", encoding="utf-8")

# docs for traceability
for doc_name in ["README.md", "PROJECT_ANALYSIS.md", "BEST_PDF_SHORTLIST.md", "FILE_DESCRIPTIONS.txt"]:
    for d in [Path(".")] + source_dirs:
        candidate = d / doc_name
        if candidate.exists():
            shutil.copy2(candidate, outdir / doc_name)
            break

# run configs / metadata
for d in source_dirs:
    if not d.exists():
        continue
    for fname in ["run_config.json", "feature_layout.json", "scaler.pkl", "label_encoder.pkl"]:
        candidate = d / fname
        if candidate.exists():
            shutil.copy2(candidate, outdir / f"{d.name}_{fname}")
            manifest.write_text(manifest.read_text(encoding="utf-8") + f"- {d.name}_{fname} <= {candidate}\n", encoding="utf-8")

print(f"[done] paper-ready results written to {outdir}")
PY
