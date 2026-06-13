# TF-QKD Intrusion Detection Project

This repository contains a complete, reproducible workflow for **Twin-Field QKD (TF-QKD)** telemetry generation, sequence-model training, anomaly detection, cross-domain evaluation, audit checks, and paper-ready export.

The project is designed for an IEEE Transactions-style study, with:
- protocol-aware synthetic TF-QKD telemetry,
- drift/asymmetry/unknown-attack variants,
- QLSTM vs LSTM vs Transformer comparison,
- one-class anomaly detection,
- cross-domain robustness testing,
- audit and paper-ready export.

## Repository layout

```text
.
├── analysis_pipeline.py
├── audit_tfqkd_outputs.py
├── generate_dataset_results.sh
├── paper_ready_export.sh
├── qkd_train_pipeline_tfqkd.py
├── tfqkd_dataset_factory.py
├── upload_ready.sh
├── data/
├── figures/
├── results/
├── paper_ready_results/
├── docs/
└── README.md
```

## What each major script does

- `tfqkd_dataset_factory.py`  
  Generates synthetic TF-QKD datasets:
  - clean
  - drift
  - asymmetry
  - unknown attack

- `qkd_train_pipeline_tfqkd.py`  
  Trains:
  - QLSTM
  - LSTM
  - Transformer
  - one-class autoencoder

- `analysis_pipeline.py`  
  Evaluates trained checkpoints, produces:
  - confusion matrices
  - ROC / PR curves
  - t-SNE embeddings
  - transformer attention
  - feature-drift plots
  - cross-domain comparison plots

- `audit_tfqkd_outputs.py`  
  Checks:
  - NaN / inf values
  - duplicates
  - zero-heavy columns
  - sequence layout consistency
  - label balance
  - metric sanity
  - prediction probability sanity
  - missing files

- `paper_ready_export.sh`  
  Flattens the key figures and tables into a single `paper_ready_results/` folder and converts plot images to PDF for LaTeX embedding.

- `upload_ready.sh`  
  Flattens the project into a single upload directory for ChatGPT Projects with intelligent file names and descriptions.

## Recommended workflow

```bash
python tfqkd_dataset_factory.py --all
python qkd_train_pipeline_tfqkd.py --data data/tfqkd_clean/tfqkd_flat.csv --outdir runs_qkd
python analysis_pipeline.py --data data/tfqkd_clean/tfqkd_flat.csv --run-dir runs_qkd --outdir analysis_clean
python analysis_pipeline.py --data data/tfqkd_drift/tfqkd_flat.csv --run-dir runs_qkd --outdir analysis_drift
python analysis_pipeline.py --data data/tfqkd_asym/tfqkd_flat.csv --run-dir runs_qkd --outdir analysis_asym
python analysis_pipeline.py --data data/tfqkd_unknown/tfqkd_flat.csv --run-dir runs_qkd --outdir analysis_unknown

python qkd_train_pipeline_tfqkd.py --data data/tfqkd_clean/tfqkd_flat.csv --outdir runs_clean_s
python qkd_train_pipeline_tfqkd.py --data data/tfqkd_drift/tfqkd_flat.csv --outdir runs_drift_s
python qkd_train_pipeline_tfqkd.py --data data/tfqkd_asym/tfqkd_flat.csv --outdir runs_asym_s
python qkd_train_pipeline_tfqkd.py --data data/tfqkd_unknown/tfqkd_flat.csv --outdir runs_unknown_s

python analysis_pipeline.py --data data/tfqkd_clean/tfqkd_flat.csv --run-dir runs_clean_s --outdir analysis_clean_s
python analysis_pipeline.py --data data/tfqkd_drift/tfqkd_flat.csv --run-dir runs_drift_s --outdir analysis_drift_s
python analysis_pipeline.py --data data/tfqkd_asym/tfqkd_flat.csv --run-dir runs_asym_s --outdir analysis_asym_s
python analysis_pipeline.py --data data/tfqkd_unknown/tfqkd_flat.csv --run-dir runs_unknown_s --outdir analysis_unknown_s

python analysis_pipeline.py --compare-dirs   analysis_clean_s analysis_drift_s analysis_asym_s analysis_unknown_s   --outdir cross_domain

python audit_tfqkd_outputs.py   --datasets data/tfqkd_clean/tfqkd_flat.csv data/tfqkd_drift/tfqkd_flat.csv data/tfqkd_asym/tfqkd_flat.csv data/tfqkd_unknown/tfqkd_flat.csv   --runs runs_qkd runs_clean_s runs_drift_s runs_asym_s runs_unknown_s   --reports analysis_clean analysis_drift analysis_asym analysis_unknown analysis_clean_s analysis_drift_s analysis_asym_s analysis_unknown_s cross_domain   --outdir audit_out

bash paper_ready_export.sh   --source-dirs .   --outdir paper_ready_results
```

## Important outputs

### Datasets
- `data/tfqkd_clean/tfqkd_flat.csv`
- `data/tfqkd_drift/tfqkd_flat.csv`
- `data/tfqkd_asym/tfqkd_flat.csv`
- `data/tfqkd_unknown/tfqkd_flat.csv`

Each dataset also keeps a long-form telemetry file:
- `tfqkd_long.csv`

### Training runs
- `runs_qkd/`
- `runs_clean_s/`
- `runs_drift_s/`
- `runs_asym_s/`
- `runs_unknown_s/`

### Analysis
- `analysis_clean/`
- `analysis_drift/`
- `analysis_asym/`
- `analysis_unknown/`
- `analysis_clean_s/`
- `analysis_drift_s/`
- `analysis_asym_s/`
- `analysis_unknown_s/`
- `cross_domain/`

### Paper-ready export
- `paper_ready_results/`

This folder is flat and intended for:
- LaTeX `\includegraphics`
- figure/table insertion
- final manuscript assembly

## Notes for IEEE-style writing

Keep the manuscript centered on:
- TF-QKD protocol realism,
- phase-lock drift and visibility,
- reference-light / wavelength sensitivity,
- unknown-attack generalization,
- cross-domain robustness,
- anomaly detection in addition to multiclass classification.

## Reproducibility reminders

Keep:
- exact commands used,
- random seeds,
- `run_config.json`,
- `feature_layout.json`,
- `scaler.pkl`,
- `label_encoder.pkl`,
- `metrics.csv`,
- `combined_metrics.csv`,
- audit reports,
- paper-ready export manifest.

