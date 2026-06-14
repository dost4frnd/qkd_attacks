# Analysis summary: tfqkd_asym

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_asym/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_asym_s`
- rows: 6400
- supervised known-label test rows: 960
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset    | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:-----------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.869792 |          0.868958 |       0.869792 |   0.869329 |            0.982071 | tfqkd_asym | qlstm       | runs_asym_s     |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.865625 |          0.866894 |       0.865625 |   0.863297 |            0.98207  | tfqkd_asym | lstm        | runs_asym_s     |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.905208 |          0.906056 |       0.905208 |   0.904624 |            0.98999  | tfqkd_asym | transformer | runs_asym_s     |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| anomaly    |   0.279167 |          0.5525   |       0.566667 |   0.277913 |          nan        | tfqkd_asym | autoencoder | runs_asym_s     |               nan |                       nan |   0.832252 |            0.960514 |         960 |           120 |           840 |     1.19796 |
