# Analysis summary: tfqkd_unknown

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_unknown/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_clean_s`
- rows: 7200
- supervised known-label test rows: 960
- unknown rows in dataset: 800

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset       | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:--------------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |    0.725   |          0.761775 |       0.725    |   0.731272 |            0.949462 | tfqkd_unknown | qlstm       | runs_clean_s    |               960 |                       800 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |    0.7875  |          0.804606 |       0.7875   |   0.793026 |            0.958566 | tfqkd_unknown | lstm        | runs_clean_s    |               960 |                       800 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| supervised |    0.89375 |          0.92422  |       0.89375  |   0.899346 |            0.991746 | tfqkd_unknown | transformer | runs_clean_s    |               960 |                       800 | nan        |          nan        |         nan |           nan |           nan |   nan       |
| anomaly    |    0.4375  |          0.545182 |       0.667276 |   0.3776   |          nan        | tfqkd_unknown | autoencoder | runs_clean_s    |               nan |                       nan |   0.821443 |            0.982003 |        1760 |           120 |          1640 |     2.20006 |
