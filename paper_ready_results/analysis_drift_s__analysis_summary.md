# Analysis summary: tfqkd_drift

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_drift/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_drift_s`
- rows: 6400
- supervised known-label test rows: 960
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset     | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:------------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.842708 |          0.837022 |       0.842708 |   0.831128 |            0.978663 | tfqkd_drift | qlstm       | runs_drift_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.85625  |          0.856791 |       0.85625  |   0.856    |            0.979859 | tfqkd_drift | lstm        | runs_drift_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.938542 |          0.93893  |       0.938542 |   0.938419 |            0.99443  | tfqkd_drift | transformer | runs_drift_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| anomaly    |   0.30625  |          0.557514 |       0.582143 |   0.302955 |          nan        | tfqkd_drift | autoencoder | runs_drift_s    |               nan |                       nan |   0.760119 |            0.946976 |         960 |           120 |           840 |     1.14607 |
