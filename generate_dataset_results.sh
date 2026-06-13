#!/usr/bin/env bash
# Reproducible TF-QKD workflow for the repository.
# Sequence:
# 1) generate datasets
# 2) train baseline on clean
# 3) evaluate baseline clean-trained model on clean/drift/asym/unknown
# 4) train per-domain model sets
# 5) evaluate per-domain model sets
# 6) cross-domain comparison
# 7) audit outputs
# 8) export paper-ready figures/tables as PDFs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_FACTORY="$SCRIPT_DIR/tfqkd_dataset_factory.py"
TRAIN_SCRIPT="$SCRIPT_DIR/qkd_train_pipeline_tfqkd.py"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analysis_pipeline.py"
AUDIT_SCRIPT="$SCRIPT_DIR/audit_tfqkd_outputs.py"
EXPORT_SCRIPT="$SCRIPT_DIR/paper_ready_export.sh"

DATASETS=(clean drift asym unknown)
DATASET_DIR="$SCRIPT_DIR/tfqkd_datasets"
BASE_RUN_DIR="$SCRIPT_DIR/runs_qkd"

make_dataset() {
  echo "[1/8] generating datasets"
  "$PYTHON_BIN" "$DATASET_FACTORY" --all
}

train_baseline_clean() {
  echo "[2/8] training baseline on clean"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --data "$DATASET_DIR/tfqkd_clean/tfqkd_flat.csv" \
    --outdir "$BASE_RUN_DIR"
}

evaluate_baseline_clean() {
  echo "[3/8] evaluating baseline clean-trained model on all datasets"
  for ds in "${DATASETS[@]}"; do
    "$PYTHON_BIN" "$ANALYSIS_SCRIPT" \
      --data "$DATASET_DIR/tfqkd_${ds}/tfqkd_flat.csv" \
      --run-dir "$BASE_RUN_DIR" \
      --outdir "$SCRIPT_DIR/analysis_${ds}"
  done
}

train_per_domain() {
  echo "[4/8] training per-domain model sets"
  for ds in "${DATASETS[@]}"; do
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
      --data "$DATASET_DIR/tfqkd_${ds}/tfqkd_flat.csv" \
      --outdir "$SCRIPT_DIR/runs_${ds}_s"
  done
}

evaluate_per_domain() {
  echo "[5/8] evaluating per-domain trained model sets"
  for ds in "${DATASETS[@]}"; do
    "$PYTHON_BIN" "$ANALYSIS_SCRIPT" \
      --data "$DATASET_DIR/tfqkd_${ds}/tfqkd_flat.csv" \
      --run-dir "$SCRIPT_DIR/runs_${ds}_s" \
      --outdir "$SCRIPT_DIR/analysis_${ds}_s"
  done
}

cross_domain_compare() {
  echo "[6/8] cross-domain comparison"
  "$PYTHON_BIN" "$ANALYSIS_SCRIPT" --compare-dirs \
    "$SCRIPT_DIR/analysis_clean_s" \
    "$SCRIPT_DIR/analysis_drift_s" \
    "$SCRIPT_DIR/analysis_asym_s" \
    "$SCRIPT_DIR/analysis_unknown_s" \
    --outdir "$SCRIPT_DIR/cross_domain"
}

audit_all() {
  echo "[7/8] auditing datasets, run dirs, and analysis dirs"
  "$PYTHON_BIN" "$AUDIT_SCRIPT" \
    --datasets \
      "$DATASET_DIR/tfqkd_clean/tfqkd_flat.csv" \
      "$DATASET_DIR/tfqkd_drift/tfqkd_flat.csv" \
      "$DATASET_DIR/tfqkd_asym/tfqkd_flat.csv" \
      "$DATASET_DIR/tfqkd_unknown/tfqkd_flat.csv" \
    --runs \
      "$BASE_RUN_DIR" \
      "$SCRIPT_DIR/runs_clean_s" \
      "$SCRIPT_DIR/runs_drift_s" \
      "$SCRIPT_DIR/runs_asym_s" \
      "$SCRIPT_DIR/runs_unknown_s" \
    --reports \
      "$SCRIPT_DIR/analysis_clean" \
      "$SCRIPT_DIR/analysis_drift" \
      "$SCRIPT_DIR/analysis_asym" \
      "$SCRIPT_DIR/analysis_unknown" \
      "$SCRIPT_DIR/analysis_clean_s" \
      "$SCRIPT_DIR/analysis_drift_s" \
      "$SCRIPT_DIR/analysis_asym_s" \
      "$SCRIPT_DIR/analysis_unknown_s" \
      "$SCRIPT_DIR/cross_domain" \
    --outdir "$SCRIPT_DIR/audit_out"
}

export_paper_ready() {
  echo "[8/8] exporting paper-ready flat results"
  bash "$EXPORT_SCRIPT" \
    --outdir "$SCRIPT_DIR/paper_ready_results" \
    --source-dirs \
      "$SCRIPT_DIR" \
    --keep-top 20
}

make_dataset
train_baseline_clean
evaluate_baseline_clean
train_per_domain
evaluate_per_domain
cross_domain_compare
audit_all
export_paper_ready

echo "[done] all workflow steps completed"
