# TF-QKD audit report

## Summary
- datasets checked: 4
- run dirs checked: 5
- report dirs checked: 9

## Dataset issues
### tfqkd_flat.csv
- rows: 6400
- cols: 388
- duplicate_rows: 0
- nan_cells: 0
- inf_cells: 0
- numeric_cols: 385
- top_zero_fraction_cols: {}
- label_counts: {'normal': 800, 'phase_drift_attack': 800, 'reference_light_tamper': 800, 'wavelength_switching_attack': 800, 'asymmetric_loss_attack': 800, 'synchronization_jitter_attack': 800, 'detector_blinding_attack': 800, 'combined_attack': 800}
- label_balance_ratio: 1.0
- split_counts: {'train': 4480, 'test': 960, 'val': 960}
- n_sequence_columns: 384
- n_bases: 12
- sequence_len: 32

### tfqkd_flat.csv
- rows: 6400
- cols: 388
- duplicate_rows: 0
- nan_cells: 0
- inf_cells: 0
- numeric_cols: 385
- top_zero_fraction_cols: {}
- label_counts: {'normal': 800, 'phase_drift_attack': 800, 'reference_light_tamper': 800, 'wavelength_switching_attack': 800, 'asymmetric_loss_attack': 800, 'synchronization_jitter_attack': 800, 'detector_blinding_attack': 800, 'combined_attack': 800}
- label_balance_ratio: 1.0
- split_counts: {'train': 4480, 'test': 960, 'val': 960}
- n_sequence_columns: 384
- n_bases: 12
- sequence_len: 32

### tfqkd_flat.csv
- rows: 6400
- cols: 388
- duplicate_rows: 0
- nan_cells: 0
- inf_cells: 0
- numeric_cols: 385
- top_zero_fraction_cols: {}
- label_counts: {'normal': 800, 'phase_drift_attack': 800, 'reference_light_tamper': 800, 'wavelength_switching_attack': 800, 'asymmetric_loss_attack': 800, 'synchronization_jitter_attack': 800, 'detector_blinding_attack': 800, 'combined_attack': 800}
- label_balance_ratio: 1.0
- split_counts: {'train': 4480, 'test': 960, 'val': 960}
- n_sequence_columns: 384
- n_bases: 12
- sequence_len: 32

### tfqkd_flat.csv
- rows: 7200
- cols: 388
- duplicate_rows: 0
- nan_cells: 0
- inf_cells: 0
- numeric_cols: 385
- top_zero_fraction_cols: {}
- label_counts: {'normal': 800, 'phase_drift_attack': 800, 'reference_light_tamper': 800, 'wavelength_switching_attack': 800, 'asymmetric_loss_attack': 800, 'synchronization_jitter_attack': 800, 'detector_blinding_attack': 800, 'combined_attack': 800, 'unknown_attack': 800}
- label_balance_ratio: 1.0
- split_counts: {'train': 4480, 'test': 1760, 'val': 960}
- n_sequence_columns: 384
- n_bases: 12
- sequence_len: 32

## Run directory issues
### /home/user/working/all_qkds/qkd_attacks/runs_qkd
- metrics.rows: 4
- metrics.cols: 10
- metrics.models: ['qlstm', 'lstm', 'transformer', 'autoencoder']

### /home/user/working/all_qkds/qkd_attacks/runs_clean_s
- metrics.rows: 4
- metrics.cols: 10
- metrics.models: ['qlstm', 'lstm', 'transformer', 'autoencoder']

### /home/user/working/all_qkds/qkd_attacks/runs_drift_s
- metrics.rows: 4
- metrics.cols: 10
- metrics.models: ['qlstm', 'lstm', 'transformer', 'autoencoder']

### /home/user/working/all_qkds/qkd_attacks/runs_asym_s
- metrics.rows: 4
- metrics.cols: 10
- metrics.models: ['qlstm', 'lstm', 'transformer', 'autoencoder']

### /home/user/working/all_qkds/qkd_attacks/runs_unknown_s
- metrics.rows: 4
- metrics.cols: 10
- metrics.models: ['qlstm', 'lstm', 'transformer', 'autoencoder']

## Analysis report issues
### /home/user/working/all_qkds/qkd_attacks/analysis_clean
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_drift
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_asym
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_unknown
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_clean_s
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_drift_s
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_asym_s
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/analysis_unknown_s
- pdf_count: 15
- csv_count: 5
- metrics rows: 4
- metrics cols: 8

### /home/user/working/all_qkds/qkd_attacks/cross_domain
- pdf_count: 7
- csv_count: 1
