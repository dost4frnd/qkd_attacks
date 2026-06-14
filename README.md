# TF-QKD Intrusion Detection Project

This repository contains a reproducible workflow for Twin-Field QKD (TF-QKD) telemetry generation, attack simulation, sequence-model training, anomaly detection, cross-domain evaluation, audit checking, and paper-ready export.

## What is included

- `tfqkd_dataset_factory.py` — generates TF-QKD datasets for clean, drift, asymmetric, and unknown-attack settings
- `qkd_train_pipeline_tfqkd.py` — trains QLSTM, LSTM, Transformer, and one-class autoencoder baselines
- `analysis_pipeline.py` — evaluates models, generates PDF figures, attention plots, drift plots, and cross-domain summaries
- `audit_tfqkd_outputs.py` — checks datasets, run folders, analysis outputs, and paper-ready inventory
- `generate_dataset_results.sh` — reproduces the full workflow
- `paper_ready_export.sh` — collects only the figures and CSVs that should be used in the paper
- `upload_ready.sh` — flattens the repository into single files for ChatGPT Project upload
- `FILE_DESCRIPTIONS.txt` — compact file inventory for later prompting

## Workflow

The intended research sequence is:

1. Generate all TF-QKD datasets
2. Train all models on all datasets
3. Evaluate same-domain results
4. Train on clean and evaluate on drift / asym / unknown
5. Compare cross-domain robustness
6. Audit all outputs
7. Export paper-ready PDFs and CSVs
8. Flatten the project for upload and writing

## Dataset variants

- `tfqkd_clean` — baseline/easy setting
- `tfqkd_drift` — stronger phase drift and synchronization instability
- `tfqkd_asym` — asymmetric channel setting
- `tfqkd_unknown` — includes unseen attacks for anomaly and generalization testing

Each dataset contains:

- `tfqkd_long.csv` / `tfqkd_long.parquet` — event-level telemetry for drift and physics plots
- `tfqkd_flat.csv` / `tfqkd_flat.parquet` — flattened sequence windows for model training

## Why long and flat files are separate

- `tfqkd_long` is used for physics interpretation and drift plots
- `tfqkd_flat` is used for sequence learning and model training

That separation is intentional and should be preserved in the paper.

## PDF-first outputs

All analysis figures are now generated as **PDF** for LaTeX embedding.

Typical file types in the analysis folders:

- confusion matrices
- supervised metric summaries
- anomaly metric summaries
- ROC / PR curves
- transformer attention heatmaps
- t-SNE embeddings
- drift traces for phase lock / QBER / visibility / reference light
- cross-domain comparison charts

## Recommended publication structure

For IEEE Transactions, keep the main paper split into:

- TF-QKD simulator and attack model
- supervised multiclass classification
- one-class anomaly detection
- cross-domain robustness
- explainability and drift analysis

Do **not** merge the one-class detector with the multiclass models into a single fairness bar chart. Treat it as a separate evaluation task.

## Main commands

Generate datasets and run the full workflow:

```bash
./generate_dataset_results.sh
```

Export only the paper-ready figures and CSVs:

```bash
./paper_ready_export.sh
```

Flatten the repository for ChatGPT Project upload:

```bash
./upload_ready.sh .
```

Run the audit separately if needed:

```bash
python audit_tfqkd_outputs.py \
  --datasets tfqkd_datasets/tfqkd_clean/tfqkd_flat.csv \
             tfqkd_datasets/tfqkd_drift/tfqkd_flat.csv \
             tfqkd_datasets/tfqkd_asym/tfqkd_flat.csv \
             tfqkd_datasets/tfqkd_unknown/tfqkd_flat.csv \
  --runs runs_clean_s runs_drift_s runs_asym_s runs_unknown_s \
  --reports analysis_clean_s analysis_drift_s analysis_asym_s analysis_unknown_s \
            analysis_clean_to_clean analysis_clean_to_drift analysis_clean_to_asym analysis_clean_to_unknown \
            cross_domain_clean_model audit_out \
  --outdir audit_out
```

## Directory conventions

- `tfqkd_datasets/` — generated datasets
- `runs_*` — training outputs and checkpoints
- `analysis_*` — per-dataset evaluation outputs and figures
- `cross_domain*` — cross-domain comparison outputs
- `paper_ready_results/` — selected figures and CSVs for the manuscript
- `audit_out/` — integrity and reproducibility checks

## Notes for paper drafting

The file `FILE_DESCRIPTIONS.txt` is designed to help later prompts by giving short descriptions of each file.  
The Markdown notes in `PROJECT_ANALYSIS.md` and `BEST_PDF_SHORTLIST.md` are intended to guide figure selection for the manuscript.

## Reproducibility checklist

Before submission, keep:
- exact command history
- `run_config.json`
- `metrics.csv`
- `figure_index.json`
- `audit_report.md` / `audit_report.json`
- `paper_ready_results/`
- the long-format data files for drift plots
- the flattened training files for model reproducibility
