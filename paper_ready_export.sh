#!/usr/bin/env bash
# =========================================================
# paper_ready_export.sh
#
# Copy ONLY paper-ready figures / tables / CSVs into a single
# flat folder for LaTeX embedding and paper drafting.
# PDF figures are preferred.
# =========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/paper_ready_results}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

copy_flat() {
  local src="$1"
  local tag="$2"
  local base
  base="$(basename "$src")"
  cp "$src" "$OUT_DIR/${tag}__${base}"
}

# Preferred PDF figure patterns
patterns=(
  "*_confusion.pdf"
  "*_supervised_metrics.pdf"
  "*_anomaly_metrics.pdf"
  "*_attention.pdf"
  "*_tsne.pdf"
  "*_drift_*.pdf"
  "*_roc.pdf"
  "*_pr.pdf"
  "*_scores.pdf"
  "cross_domain*.pdf"
)

# Preferred tabular / summary outputs
csv_patterns=(
  "metrics.csv"
  "metrics_supervised.csv"
  "metrics_anomaly.csv"
  "combined_metrics.csv"
)

md_patterns=(
  "analysis_summary.md"
  "audit_report.md"
  "README.md"
  "PROJECT_ANALYSIS.md"
  "BEST_PDF_SHORTLIST.md"
)

json_patterns=(
  "figure_index.json"
  "audit_report.json"
)

# scan the key directories only
scan_dirs=(
  "$ROOT_DIR/analysis_clean"
  "$ROOT_DIR/analysis_drift"
  "$ROOT_DIR/analysis_asym"
  "$ROOT_DIR/analysis_unknown"
  "$ROOT_DIR/analysis_clean_s"
  "$ROOT_DIR/analysis_drift_s"
  "$ROOT_DIR/analysis_asym_s"
  "$ROOT_DIR/analysis_unknown_s"
  "$ROOT_DIR/analysis_clean_to_clean"
  "$ROOT_DIR/analysis_clean_to_drift"
  "$ROOT_DIR/analysis_clean_to_asym"
  "$ROOT_DIR/analysis_clean_to_unknown"
  "$ROOT_DIR/cross_domain"
  "$ROOT_DIR/cross_domain_clean_model"
  "$ROOT_DIR/audit_out"
)

echo "[+] exporting paper-ready artifacts to $OUT_DIR"

for d in "${scan_dirs[@]}"; do
  [ -d "$d" ] || continue
  tag="$(basename "$d")"
  for pat in "${patterns[@]}"; do
    while IFS= read -r f; do
      [ -f "$f" ] && copy_flat "$f" "$tag"
    done < <(find "$d" -maxdepth 1 -type f -name "$pat" | sort)
  done
  for pat in "${csv_patterns[@]}"; do
    while IFS= read -r f; do
      [ -f "$f" ] && copy_flat "$f" "$tag"
    done < <(find "$d" -maxdepth 1 -type f -name "$pat" | sort)
  done
  for pat in "${md_patterns[@]}"; do
    while IFS= read -r f; do
      [ -f "$f" ] && copy_flat "$f" "$tag"
    done < <(find "$d" -maxdepth 1 -type f -name "$pat" | sort)
  done
  for pat in "${json_patterns[@]}"; do
    while IFS= read -r f; do
      [ -f "$f" ] && copy_flat "$f" "$tag"
    done < <(find "$d" -maxdepth 1 -type f -name "$pat" | sort)
  done
done

# add a small index for TeX drafting
{
  echo "# Paper-ready export index"
  echo
  echo "Copied files:"
  find "$OUT_DIR" -maxdepth 1 -type f | sort | sed 's#^#- #'
} > "$OUT_DIR/PAPER_READY_INDEX.md"

echo "[+] done"
echo "    $OUT_DIR"
