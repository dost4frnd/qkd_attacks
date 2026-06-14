# Best PDF shortlist for the paper

This list is intentionally narrow. These are the figures that best support the manuscript story.

## Must include in the main text

1. `*_drift_phase_lock_error_rad.pdf`  
   Why: shows physical lock instability and attack sensitivity.

2. `*_drift_qber_phase.pdf`  
   Why: shows security degradation directly.

3. `*_transformer_attention.pdf`  
   Why: demonstrates explainability and temporal focus.

4. `*_transformer_confusion.pdf`  
   Why: strongest supervised baseline comparison.

5. `*_qlstm_confusion.pdf`  
   Why: compares the quantum-inspired baseline.

6. `*_lstm_confusion.pdf`  
   Why: classical temporal baseline.

7. `*_supervised_metrics.pdf`  
   Why: compact summary for the three supervised models.

8. `*_anomaly_metrics.pdf`  
   Why: separate one-class detector summary.

9. `*_autoencoder_roc.pdf`  
   Why: anomaly-detection quality.

10. `*_autoencoder_pr.pdf`  
    Why: useful under class imbalance / unknown attacks.

11. `*_autoencoder_scores.pdf`  
    Why: reconstruction error distribution.

12. `*_tsne.pdf`  
    Why: latent-space separation illustration.

13. `cross_domain_supervised_accuracy.pdf`  
    Why: domain-shift summary.

14. `cross_domain_supervised_f1_macro.pdf`  
    Why: class-sensitive robustness summary.

15. `cross_domain_supervised_roc_auc_ovr_macro.pdf`  
    Why: ranking quality under domain shift.

16. `cross_domain_anomaly_roc_auc.pdf`  
    Why: zero-day robustness.

17. `cross_domain_anomaly_average_precision.pdf`  
    Why: precision under attack imbalance.

18. `cross_domain_anomaly_f1_macro.pdf`  
    Why: thresholded anomaly performance.

19. `cross_domain_anomaly_accuracy.pdf`  
    Why: complementary anomaly summary.

20. `cross_domain*.pdf` (any remaining)  
    Why: overall manuscript robustness view.

## Supplementary-only if space is tight

- extra drift traces for reference light
- extra drift traces for visibility
- additional per-class confusion matrices for every dataset
- full long-table drift figures for all features

## Practical rule

If a figure does not help answer one of these questions, it should likely move to the supplement:

- does the model detect attacks?
- does it generalize across domains?
- does the transformer focus on the right time steps?
- does phase-lock drift align with QBER degradation?
