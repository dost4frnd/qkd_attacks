#!/usr/bin/env bash
# Flatten the project into one upload folder for ChatGPT Projects.
# No subdirectories are created in the output.

set -euo pipefail

SRC="${1:-.}"
OUT="flat_upload"
DESC_FILE="$OUT/FILE_DESCRIPTIONS.txt"

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$DESC_FILE"

counter=0

sanitize_name() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//[^[:alnum:]_.-]/_}"
  printf '%s' "$s"
}

add_file() {
  local srcfile="$1"
  local category="$2"
  local desc="$3"
  local base cleanbase newname
  base="$(basename "$srcfile")"
  cleanbase="$(sanitize_name "$base")"
  newname="${category}_$(printf '%04d' "$counter")_${cleanbase}"
  cp "$srcfile" "$OUT/$newname"
  {
    echo "FILE: $newname"
    echo "TYPE: $category"
    echo "SOURCE: $srcfile"
    echo "DESCRIPTION: $desc"
    echo
  } >> "$DESC_FILE"
  counter=$((counter + 1))
}

should_skip() {
  local f="$1"
  case "$f" in
    */__pycache__/*|*/flat_upload/*|*/paper_ready_results/*|*/tfqkd_repo_v2/*)
      return 0
      ;;
    *.zip|*.tar.gz|*.tgz|*.gz|*.bz2|*.xz)
      return 0
      ;;
  esac
  return 1
}

echo "[+] Flattening: $SRC -> $OUT"

while IFS= read -r -d '' f; do
  should_skip "$f" && continue
  lower="$(printf '%s' "$f" | tr '[:upper:]' '[:lower:]')"

  case "$lower" in
    *.csv|*.parquet)
      case "$lower" in
        *clean*)    add_file "$f" "DATASET_CLEAN" "Clean TF-QKD telemetry dataset for baseline training and evaluation." ;;
        *drift*)    add_file "$f" "DATASET_DRIFT" "TF-QKD dataset with phase drift and synchronization instability." ;;
        *asym*)     add_file "$f" "DATASET_ASYM" "TF-QKD dataset with asymmetric channel conditions." ;;
        *unknown*)   add_file "$f" "DATASET_UNKNOWN" "TF-QKD dataset with unseen attack behavior for anomaly testing." ;;
        *metrics*)   add_file "$f" "TABLE" "Metrics table or combined results CSV from training/evaluation." ;;
        *manifest*)  add_file "$f" "REPORT" "Manifest or file inventory for reproducibility." ;;
        *)           add_file "$f" "DATASET_MISC" "General dataset or telemetry file for TF-QKD experiments." ;;
      esac
      ;;
    *.py)
      case "$lower" in
        *train*)     add_file "$f" "SCRIPT" "Training pipeline for TF-QKD intrusion detection experiments." ;;
        *analysis*)   add_file "$f" "SCRIPT" "Analysis and evaluation script generating figures and metrics." ;;
        *dataset*)    add_file "$f" "SCRIPT" "Synthetic TF-QKD telemetry dataset generation script." ;;
        *audit*)     add_file "$f" "SCRIPT" "Audit and integrity validation script for datasets and results." ;;
        *)           add_file "$f" "SCRIPT" "Research source code related to TF-QKD AI experiments." ;;
      esac
      ;;
    *.ipynb)
      add_file "$f" "NOTEBOOK" "Jupyter notebook containing exploratory analysis or experiments."
      ;;
    *.pdf)
      case "$lower" in
        *confusion*) add_file "$f" "FIGURE" "Confusion matrix visualization for classification results." ;;
        *roc*)       add_file "$f" "FIGURE" "ROC curve showing classifier discrimination performance." ;;
        *pr*)        add_file "$f" "FIGURE" "Precision-Recall curve for intrusion detection evaluation." ;;
        *attention*) add_file "$f" "FIGURE" "Transformer attention visualization over TF-QKD telemetry." ;;
        *tsne*)      add_file "$f" "FIGURE" "t-SNE latent embedding visualization of telemetry features." ;;
        *drift*)     add_file "$f" "FIGURE" "Feature drift visualization for TF-QKD telemetry." ;;
        *metrics*)   add_file "$f" "FIGURE" "Comparison chart for model metrics or cross-domain results." ;;
        *)           add_file "$f" "FIGURE" "Publication figure generated from analysis or evaluation." ;;
      esac
      ;;
    *.png|*.jpg|*.jpeg|*.svg)
      case "$lower" in
        *confusion*) add_file "$f" "FIGURE_RAW" "Raster confusion matrix visualization from prior run." ;;
        *roc*)       add_file "$f" "FIGURE_RAW" "Raster ROC curve from prior run." ;;
        *pr*)        add_file "$f" "FIGURE_RAW" "Raster PR curve from prior run." ;;
        *attention*) add_file "$f" "FIGURE_RAW" "Raster transformer attention visualization from prior run." ;;
        *tsne*)      add_file "$f" "FIGURE_RAW" "Raster t-SNE embedding plot from prior run." ;;
        *)           add_file "$f" "FIGURE_RAW" "Raster figure from prior run." ;;
      esac
      ;;
    *.md)
      case "$lower" in
        *readme*)       add_file "$f" "REPORT" "Project README and workflow overview." ;;
        *analysis*)     add_file "$f" "REPORT" "Project analysis and execution summary." ;;
        *project*)      add_file "$f" "REPORT" "Project overview and research plan." ;;
        *shortlist*)    add_file "$f" "REPORT" "Shortlist of paper-ready figures." ;;
        *manifest*)     add_file "$f" "REPORT" "Manifest or index file for generated artifacts." ;;
        *audit*)        add_file "$f" "REPORT" "Audit report for dataset and result integrity." ;;
        *)              add_file "$f" "REPORT" "Markdown research note or report." ;;
      esac
      ;;
    *.txt)
      case "$lower" in
        *workflow*)     add_file "$f" "REPORT" "Workflow notes describing the experiment sequence." ;;
        *sources*)      add_file "$f" "REFERENCES" "Source mapping for paper writing and literature review." ;;
        *template*)     add_file "$f" "REFERENCES" "IEEE template notes and resources." ;;
        *paper*)        add_file "$f" "REFERENCES" "Paper-related source notes or reference mapping." ;;
        *)              add_file "$f" "REPORT" "Text note or source file for the research project." ;;
      esac
      ;;
    *.bib)
      add_file "$f" "REFERENCES" "BibTeX references for manuscript writing."
      ;;
  esac
done < <(find "$SRC" -type f -print0)

echo ""
echo "[+] DONE"
echo "[+] Flattened upload directory:"
echo "    $OUT"
echo ""
echo "[+] Description file:"
echo "    $DESC_FILE"
echo ""
echo "[+] Upload ALL files from:"
echo "    $OUT"
echo ""
echo "[+] No subdirectories remain."
