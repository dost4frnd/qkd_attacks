#!/usr/bin/env bash
# =========================================================
# upload_ready.sh
#
# Flatten the repo into single files for ChatGPT Project upload.
# No subdirectories in the final upload folder.
# Includes PDF figures, CSVs, scripts, and docs.
# =========================================================

set -euo pipefail

SRC="${1:-.}"
OUT="flat_upload"

rm -rf "$OUT"
mkdir -p "$OUT"

DESC_FILE="$OUT/FILE_DESCRIPTIONS.txt"
echo "# FILE DESCRIPTIONS FOR PAPER WRITING" > "$DESC_FILE"
echo "" >> "$DESC_FILE"

counter=0

copy_and_rename () {
    local srcfile="$1"
    local category="$2"
    local desc="$3"
    local base cleanbase newname

    base="$(basename "$srcfile")"
    cleanbase="$(echo "$base" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
    newname="${category}_$(printf "%04d" "$counter")_${cleanbase}"

    cp "$srcfile" "$OUT/$newname"

    {
      echo "FILE: $newname"
      echo "TYPE: $category"
      echo "DESCRIPTION: $desc"
      echo
    } >> "$DESC_FILE"

    counter=$((counter+1))
}

echo "[+] scanning files under $SRC"

find "$SRC" -type f | while IFS= read -r f; do
    lower="$(echo "$f" | tr '[:upper:]' '[:lower:]')"

    # skip generated archives / upload folders
    case "$lower" in
      *flat_upload/*|*tfqkd_repo_v3_bundle.zip|*project_docs.tar.gz|*tfqkd_repo_bundle.zip|*tfqkd_repo_v2_bundle.zip)
        continue
        ;;
    esac

    if [[ "$lower" == *.csv ]] || [[ "$lower" == *.parquet ]]; then
        if [[ "$lower" == *clean* ]]; then
            desc="Clean TF-QKD telemetry dataset for baseline training and testing."
            copy_and_rename "$f" "DATASET_CLEAN" "$desc"
        elif [[ "$lower" == *drift* ]]; then
            desc="TF-QKD dataset with stronger phase drift and synchronization instability."
            copy_and_rename "$f" "DATASET_DRIFT" "$desc"
        elif [[ "$lower" == *asym* ]]; then
            desc="TF-QKD dataset with asymmetric channel conditions."
            copy_and_rename "$f" "DATASET_ASYM" "$desc"
        elif [[ "$lower" == *unknown* ]]; then
            desc="TF-QKD unseen-attack dataset for anomaly and generalization testing."
            copy_and_rename "$f" "DATASET_UNKNOWN" "$desc"
        else
            desc="General dataset or metrics table for the TF-QKD project."
            copy_and_rename "$f" "DATASET_MISC" "$desc"
        fi

    elif [[ "$lower" == *.py ]]; then
        if [[ "$lower" == *train* ]]; then
            desc="Model training pipeline for QKD intrusion detection."
        elif [[ "$lower" == *analysis* ]]; then
            desc="Analysis pipeline generating PDF plots, metrics, and comparisons."
        elif [[ "$lower" == *dataset* ]]; then
            desc="Synthetic TF-QKD telemetry dataset generation script."
        elif [[ "$lower" == *audit* ]]; then
            desc="Audit script for dataset, metrics, and figure integrity."
        else
            desc="Research source code related to TF-QKD experiments."
        fi
        copy_and_rename "$f" "SCRIPT" "$desc"

    elif [[ "$lower" == *.sh ]]; then
        desc="Shell workflow script for dataset generation, evaluation, export, or upload."
        copy_and_rename "$f" "SHELL" "$desc"

    elif [[ "$lower" == *.ipynb ]]; then
        desc="Notebook for exploratory TF-QKD analysis."
        copy_and_rename "$f" "NOTEBOOK" "$desc"

    elif [[ "$lower" == *.pdf ]]; then
        if [[ "$lower" == *confusion* ]]; then
            desc="Confusion matrix figure for classifier results."
        elif [[ "$lower" == *roc* ]]; then
            desc="ROC figure for model discrimination."
        elif [[ "$lower" == *pr* ]]; then
            desc="Precision-recall figure for anomaly detection or class imbalance."
        elif [[ "$lower" == *attention* ]]; then
            desc="Transformer attention figure over TF-QKD telemetry."
        elif [[ "$lower" == *tsne* ]]; then
            desc="Latent-space t-SNE figure."
        elif [[ "$lower" == *drift* ]]; then
            desc="TF-QKD drift / physics trend figure."
        else
            desc="Research figure or document in PDF format."
        fi
        copy_and_rename "$f" "FIGURE" "$desc"

    elif [[ "$lower" == *.png ]] || [[ "$lower" == *.jpg ]] || [[ "$lower" == *.jpeg ]] || [[ "$lower" == *.svg ]]; then
        if [[ "$lower" == *confusion* ]]; then
            desc="Image figure for classifier results."
        elif [[ "$lower" == *roc* ]]; then
            desc="ROC curve image."
        elif [[ "$lower" == *pr* ]]; then
            desc="Precision-recall curve image."
        elif [[ "$lower" == *attention* ]]; then
            desc="Attention heatmap image."
        elif [[ "$lower" == *tsne* ]]; then
            desc="t-SNE latent embedding image."
        else
            desc="Research figure image."
        fi
        copy_and_rename "$f" "FIGURE" "$desc"

    elif [[ "$lower" == *.txt ]] || [[ "$lower" == *.md ]]; then
        if [[ "$lower" == *cmd* ]]; then
            desc="Command history / workflow reconstruction."
        elif [[ "$lower" == *audit* ]]; then
            desc="Audit or integrity report."
        elif [[ "$lower" == *readme* ]]; then
            desc="Project documentation and workflow notes."
        else
            desc="Research note or short report."
        fi
        copy_and_rename "$f" "REPORT" "$desc"

    elif [[ "$lower" == *.json ]] || [[ "$lower" == *.pkl ]] || [[ "$lower" == *.ckpt ]] || [[ "$lower" == *.pth ]] || [[ "$lower" == *.pt ]]; then
        desc="Model artifact, configuration, or serialized experiment state."
        copy_and_rename "$f" "ARTIFACT" "$desc"
    fi
done

echo
echo "[+] done"
echo "[+] flattened upload folder: $OUT"
echo "[+] descriptions: $DESC_FILE"
