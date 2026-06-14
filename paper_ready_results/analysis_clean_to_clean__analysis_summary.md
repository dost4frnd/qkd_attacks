# Analysis summary: tfqkd_clean

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_clean/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_clean_s`
- rows: 6400
- supervised known-label test rows: 960
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset     | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:------------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.873958 |          0.872334 |       0.873958 |   0.872428 |            0.982186 | tfqkd_clean | qlstm       | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.876042 |          0.874876 |       0.876042 |   0.874319 |            0.981029 | tfqkd_clean | lstm        | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |   0.998958 |          0.998967 |       0.998958 |   0.998958 |            1        | tfqkd_clean | transformer | runs_clean_s    |               960 |                         0 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| anomaly    |   0.247917 |          0.557879 |       0.559524 |   0.247904 |          nan        | tfqkd_clean | autoencoder | runs_clean_s    |               nan |                       nan |   0.854772 |            0.969505 |         960 |           120 |           840 |     1.37727 |
