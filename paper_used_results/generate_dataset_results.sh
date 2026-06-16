#!/usr/bin/env bash
# =========================================================
# generate_dataset_results.sh
#
# End-to-end TF-QKD research workflow:
# 1) generate datasets
# 2) train all models on all datasets
# 3) evaluate same-domain
# 4) evaluate clean-trained model on other domains
# 5) cross-domain comparison
# 6) audit outputs
# 7) export paper-ready results
#
# PDF-first analysis outputs are produced by analysis_pipeline.py.
# =========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/tfqkd_datasets}"
RUN_CLEAN="${RUN_CLEAN:-$ROOT_DIR/runs_clean_s}"
RUN_DRIFT="${RUN_DRIFT:-$ROOT_DIR/runs_drift_s}"
RUN_ASYM="${RUN_ASYM:-$ROOT_DIR/runs_asym_s}"
RUN_UNKNOWN="${RUN_UNKNOWN:-$ROOT_DIR/runs_unknown_s}"

AN_CLEAN="${AN_CLEAN:-$ROOT_DIR/analysis_clean_s}"
AN_DRIFT="${AN_DRIFT:-$ROOT_DIR/analysis_drift_s}"
AN_ASYM="${AN_ASYM:-$ROOT_DIR/analysis_asym_s}"
AN_UNKNOWN="${AN_UNKNOWN:-$ROOT_DIR/analysis_unknown_s}"

AN_CLEAN_TO_CLEAN="${AN_CLEAN_TO_CLEAN:-$ROOT_DIR/analysis_clean_to_clean}"
AN_CLEAN_TO_DRIFT="${AN_CLEAN_TO_DRIFT:-$ROOT_DIR/analysis_clean_to_drift}"
AN_CLEAN_TO_ASYM="${AN_CLEAN_TO_ASYM:-$ROOT_DIR/analysis_clean_to_asym}"
AN_CLEAN_TO_UNKNOWN="${AN_CLEAN_TO_UNKNOWN:-$ROOT_DIR/analysis_clean_to_unknown}"

CROSS_DIR="${CROSS_DIR:-$ROOT_DIR/cross_domain_clean_model}"
AUDIT_DIR="${AUDIT_DIR:-$ROOT_DIR/audit_out}"
PAPER_DIR="${PAPER_DIR:-$ROOT_DIR/paper_ready_results}"

mkdir -p "$DATA_DIR" "$AUDIT_DIR"

# echo "[1/7] generating datasets"
# "$PYTHON" "$ROOT_DIR/tfqkd_dataset_factory.py" --all --outdir "$DATA_DIR"

# echo "[2/7] training all models on all datasets"
# "$PYTHON" "$ROOT_DIR/qkd_train_pipeline_tfqkd.py" --data "$DATA_DIR/tfqkd_clean/tfqkd_flat.csv"   --outdir "$RUN_CLEAN"
# "$PYTHON" "$ROOT_DIR/qkd_train_pipeline_tfqkd.py" --data "$DATA_DIR/tfqkd_drift/tfqkd_flat.csv"   --outdir "$RUN_DRIFT"
# "$PYTHON" "$ROOT_DIR/qkd_train_pipeline_tfqkd.py" --data "$DATA_DIR/tfqkd_asym/tfqkd_flat.csv"    --outdir "$RUN_ASYM"
# "$PYTHON" "$ROOT_DIR/qkd_train_pipeline_tfqkd.py" --data "$DATA_DIR/tfqkd_unknown/tfqkd_flat.csv" --outdir "$RUN_UNKNOWN"

echo "[3/7] same-domain evaluation and figure generation"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_clean/tfqkd_flat.csv"   --run-dir "$RUN_CLEAN"   --outdir "$AN_CLEAN"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_drift/tfqkd_flat.csv"   --run-dir "$RUN_DRIFT"   --outdir "$AN_DRIFT"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_asym/tfqkd_flat.csv"    --run-dir "$RUN_ASYM"    --outdir "$AN_ASYM"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_unknown/tfqkd_flat.csv" --run-dir "$RUN_UNKNOWN" --outdir "$AN_UNKNOWN"

echo "[4/7] clean-trained model evaluated on other domains"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_clean/tfqkd_flat.csv"   --run-dir "$RUN_CLEAN" --outdir "$AN_CLEAN_TO_CLEAN"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_drift/tfqkd_flat.csv"   --run-dir "$RUN_CLEAN" --outdir "$AN_CLEAN_TO_DRIFT"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_asym/tfqkd_flat.csv"    --run-dir "$RUN_CLEAN" --outdir "$AN_CLEAN_TO_ASYM"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --data "$DATA_DIR/tfqkd_unknown/tfqkd_flat.csv" --run-dir "$RUN_CLEAN" --outdir "$AN_CLEAN_TO_UNKNOWN"

echo "[5/7] cross-domain comparison"
"$PYTHON" "$ROOT_DIR/analysis_pipeline.py" --compare-dirs \
  "$AN_CLEAN_TO_CLEAN" \
  "$AN_CLEAN_TO_DRIFT" \
  "$AN_CLEAN_TO_ASYM" \
  "$AN_CLEAN_TO_UNKNOWN" \
  --outdir "$CROSS_DIR"

echo "[6/7] audit datasets, runs, and analysis outputs"
"$PYTHON" "$ROOT_DIR/audit_tfqkd_outputs.py" \
  --datasets \
    "$DATA_DIR/tfqkd_clean/tfqkd_flat.csv" \
    "$DATA_DIR/tfqkd_drift/tfqkd_flat.csv" \
    "$DATA_DIR/tfqkd_asym/tfqkd_flat.csv" \
    "$DATA_DIR/tfqkd_unknown/tfqkd_flat.csv" \
  --runs \
    "$RUN_CLEAN" \
    "$RUN_DRIFT" \
    "$RUN_ASYM" \
    "$RUN_UNKNOWN" \
  --reports \
    "$AN_CLEAN" \
    "$AN_DRIFT" \
    "$AN_ASYM" \
    "$AN_UNKNOWN" \
    "$AN_CLEAN_TO_CLEAN" \
    "$AN_CLEAN_TO_DRIFT" \
    "$AN_CLEAN_TO_ASYM" \
    "$AN_CLEAN_TO_UNKNOWN" \
    "$CROSS_DIR" \
  --outdir "$AUDIT_DIR"

echo "[7/7] export paper-ready flat results"
"$ROOT_DIR/paper_ready_export.sh"

echo
echo "[done] workflow complete"
echo " - datasets:      $DATA_DIR"
echo " - runs:          runs_*"
echo " - analyses:      analysis_* / cross_domain_clean_model"
echo " - audit:         $AUDIT_DIR"
echo " - paper-ready:   $PAPER_DIR"
