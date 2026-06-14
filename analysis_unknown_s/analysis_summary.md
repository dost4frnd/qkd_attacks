# Analysis summary: tfqkd_unknown

- dataset: `/home/user/working/all_qkds/qkd_attacks/tfqkd_datasets/tfqkd_unknown/tfqkd_flat.csv`
- run dir: `/home/user/working/all_qkds/qkd_attacks/runs_unknown_s`
- rows: 7200
- supervised known-label test rows: 1760
- unknown rows in dataset: 0

## Metrics

| task       |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_ovr_macro | dataset       | model       | train_run_dir   |   test_known_rows |   unknown_rows_in_dataset |    roc_auc |   average_precision |   test_rows |   normal_rows |   attack_rows |   threshold |
|:-----------|-----------:|------------------:|---------------:|-----------:|--------------------:|:--------------|:------------|:----------------|------------------:|--------------------------:|-----------:|--------------------:|------------:|--------------:|--------------:|------------:|
| supervised |   0.46875  |          0.582379 |       0.763889 |   0.6203   |            0.959464 | tfqkd_unknown | qlstm       | runs_unknown_s  |              1760 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| supervised |   0.473864 |          0.611062 |       0.772222 |   0.642113 |            0.947224 | tfqkd_unknown | lstm        | runs_unknown_s  |              1760 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| supervised |   0.488636 |          0.62485  |       0.796296 |   0.657229 |            0.955817 | tfqkd_unknown | transformer | runs_unknown_s  |              1760 |                         0 | nan        |          nan        |         nan |           nan |           nan |    nan      |
| anomaly    |   0.471023 |          0.553883 |       0.704573 |   0.402765 |          nan        | tfqkd_unknown | autoencoder | runs_unknown_s  |               nan |                       nan |   0.874329 |            0.988402 |        1760 |           120 |          1640 |      1.2699 |
