# Analysis summary: tfqkd_asym

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_asym/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_clean_s`
- rows: 6400
- supervised known-label test rows: 960
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset    | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:-----------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.728125 |          0.805618 |       0.728125 |   0.749737 |            0.952573 | tfqkd_asym | qlstm       | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.745833 |          0.803952 |       0.745833 |   0.767493 |            0.952191 | tfqkd_asym | lstm        | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.854167 |          0.918758 |       0.854167 |   0.868571 |            0.989036 | tfqkd_asym | transformer | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| anomaly    |   0.223958 |          0.543442 |       0.53869  |   0.223769 |          nan        | tfqkd_asym | autoencoder | runs_clean_s    |               nan |                       nan |   0.732034 |            0.935995 |         960 |           120 |           840 |     2.86816 |
