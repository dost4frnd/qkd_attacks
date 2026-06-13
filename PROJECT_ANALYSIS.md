# TF-QKD Project Analysis

## Current workflow

### 1) Dataset generation
Source:
- `tfqkd_dataset_factory.py`

Variants:
- clean
- drift
- asymmetry
- unknown attack

Outputs:
- `tfqkd_long.csv`
- `tfqkd_flat.csv`
- `config.json`
- `label_map.json`
- `manifest.json`

### 2) Training
Source:
- `qkd_train_pipeline_tfqkd.py`

Models:
- QLSTM
- LSTM
- Transformer
- one-class autoencoder

Artifacts:
- checkpoints
- scaler
- label encoder
- feature layout
- metrics
- optional predictions

### 3) Evaluation and analysis
Source:
- `analysis_pipeline.py`

Outputs:
- confusion matrices
- ROC and PR curves
- t-SNE embeddings
- transformer attention
- feature drift plots
- cross-domain comparisons

### 4) Audit
Source:
- `audit_tfqkd_outputs.py`

Checks:
- missing files
- NaN / inf values
- duplicates
- label balance
- sequence layout consistency
- prediction probability sanity
- metric bounds

### 5) Paper-ready export
Source:
- `paper_ready_export.sh`

Purpose:
- flatten the important figures/tables
- convert plots to PDF
- create a LaTeX-ready asset set

### 6) Project upload flattening
Source:
- `upload_ready.sh`

Purpose:
- flatten the whole project into a single upload folder
- add compact descriptions for later prompting

## Scientific emphasis

The strongest part of the study is the **cross-domain TF-QKD robustness question**:

- train on clean
- evaluate on drift
- evaluate on asymmetry
- evaluate on unknown attack

This directly tests whether the models learn TF-QKD physics rather than memorizing a single synthetic distribution.

## Suggested paper contribution framing

1. Physics-informed TF-QKD telemetry generator
2. Multimodel comparison:
   - QLSTM
   - LSTM
   - Transformer
   - one-class anomaly detection
3. Domain shift and unknown-attack robustness
4. Explainability through transformer attention and drift plots
5. Audit-friendly reproducibility package

## Practical notes often forgotten

- keep the raw commands
- keep the flat outputs and long-form telemetry both
- keep PDF plots for LaTeX
- keep the audit report
- keep the exact export manifest
- keep the dataset manifest and configuration files
- keep cross-domain results and not only the best run

