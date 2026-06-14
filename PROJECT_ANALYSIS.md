# Project Analysis and Figure Strategy

## Core assessment

The project direction is strong for an IEEE Transactions-style paper because it separates three distinct tasks:

1. **Physics-informed TF-QKD telemetry generation**
2. **Supervised multiclass intrusion detection**
3. **One-class / anomaly detection for unknown attacks**

That separation is important because the metrics are not identical.  
A multiclass classifier should not be plotted on the same bar chart as a one-class detector without clear labeling and task separation.

## What is scientifically strong already

- balanced baseline datasets are good for debugging and method comparison
- the drift and asymmetry variants give physically meaningful perturbations
- the unknown-attack dataset is useful for zero-day style anomaly testing
- cross-domain evaluation is much stronger than in-domain evaluation alone
- attention visualization adds interpretability

## What needed correction

### 1. Supervised vs anomaly metrics
The autoencoder must be reported separately from QLSTM / LSTM / Transformer.

Recommended split:
- `supervised_metrics.pdf`
- `anomaly_metrics.pdf`

### 2. PDF output
All main figures should be PDF, not PNG, to simplify LaTeX embedding.

### 3. Cross-domain tests
The strongest claim comes from:
- train on clean
- test on drift
- test on asym
- test on unknown

### 4. Unknown attacks
Unknown labels should be treated as out-of-distribution for anomaly detection, not forced into the supervised fairness comparison.

## Figure policy

### Main-text candidates
- drift phase-lock error
- drift QBER phase
- transformer attention
- supervised confusion matrices
- supervised metric summary
- anomaly ROC / PR
- cross-domain comparison plots

### Supplementary candidates
- t-SNE embeddings
- extra per-feature drift plots
- score histograms
- long figure variants

## What the current pipeline now does

- produces per-dataset PDF figures
- stores a `figure_index.json`
- keeps metrics in separate supervised/anomaly rows
- supports clean-trained cross-domain evaluation
- creates cross-domain comparison charts from analysis outputs

## Recommended manuscript angle

The paper should be framed as:

> Physics-informed TF-QKD telemetry anomaly detection with recurrent, attention-based, and one-class models under domain shift.

That is much stronger than a generic “ML on QKD” framing.
