# Analysis summary: tfqkd_drift

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_drift/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_clean_s`
- rows: 6400
- supervised known-label test rows: 960
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset     | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:------------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.725    |          0.756258 |       0.725    |   0.711857 |            0.956048 | tfqkd_drift | qlstm       | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| supervised |   0.772917 |          0.790723 |       0.772917 |   0.767981 |            0.949121 | tfqkd_drift | lstm        | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| supervised |   0.927083 |          0.931891 |       0.927083 |   0.92621  |            0.988207 | tfqkd_drift | transformer | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| anomaly    |   0.292708 |          0.548921 |       0.567262 |   0.290206 |          nan        | tfqkd_drift | autoencoder | runs_clean_s    |               nan |                       nan |   0.739802 |            0.941649 |         960 |           120 |           840 |      2.4266 |
