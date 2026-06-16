import os
import zipfile

# Dictionary containing the independent scripts for each table
table_scripts = {
    "table1_dataset_parameters.py": '''import pandas as pd

def generate_table():
    """Generates Table 1: TF-QKD Dataset Variations and Attack Distributions"""
    data = [
        {"Dataset": "tfqkd_clean", "Samples": 6400, "Noise Profile": "Standard", "Asym Loss (dB)": 0, "Phase Drift": "None"},
        {"Dataset": "tfqkd_drift", "Samples": 6400, "Noise Profile": "Ornstein-Uhlenbeck", "Asym Loss (dB)": 0, "Phase Drift": "High"},
        {"Dataset": "tfqkd_asym", "Samples": 6400, "Noise Profile": "Standard", "Asym Loss (dB)": 3, "Phase Drift": "None"},
        {"Dataset": "tfqkd_unknown", "Samples": 7200, "Noise Profile": "Mixed", "Asym Loss (dB)": "0-3", "Phase Drift": "Variable"}
    ]
    df = pd.DataFrame(data)
    
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{TF-QKD Dataset Variations and Attack Distributions}\\n\\\\label{tab:dataset_params}\\n"
    latex += "\\\\begin{tabular}{@{}lcccc@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Dataset Variant} & \\\\textbf{Total Samples} & \\\\textbf{Phase Drift Noise} & \\\\textbf{Asymmetric Loss} \\\\\\\\ \\\\midrule\\n"
    
    for _, row in df.iterrows():
        latex += f"{row['Dataset'].replace('_', '\\\\_')} & {row['Samples']} & {row['Noise Profile']} & {row['Asym Loss (dB)']} \\\\\\\\\\n"
    
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table1_dataset_parameters.tex", "w") as f:
        f.write(latex)
    print("Table 1 generated successfully: table1_dataset_parameters.tex")

if __name__ == "__main__":
    generate_table()
''',

    "table2_clean_supervised.py": '''import pandas as pd

def generate_table(csv_path="analysis_clean_s__metrics_supervised.csv"):
    """Generates Table 2: Same-Domain Supervised Performance (CLEAN)"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{Same-Domain Supervised Performance Metrics (CLEAN)}\\n\\\\label{tab:clean_metrics}\\n"
    latex += "\\\\begin{tabular}{@{}lcccc@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Model} & \\\\textbf{Accuracy} & \\\\textbf{Precision} & \\\\textbf{Recall} & \\\\textbf{F1-Score} \\\\\\\\ \\\\midrule\\n"
    
    for _, row in df.iterrows():
        latex += f"{row['model'].upper()} & {row['accuracy']:.4f} & {row['precision_macro']:.4f} & {row['recall_macro']:.4f} & {row['f1_macro']:.4f} \\\\\\\\\\n"
    
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table2_clean_supervised.tex", "w") as f:
        f.write(latex)
    print("Table 2 generated successfully: table2_clean_supervised.tex")

if __name__ == "__main__":
    generate_table()
''',

    "table3_drift_robustness.py": '''import pandas as pd

def generate_table(csv_path="analysis_drift_s__metrics_supervised.csv"):
    """Generates Table 3: Robustness to Environmental Phase Drift"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{Model Robustness under Environmental Phase Drift (DRIFT)}\\n\\\\label{tab:drift_metrics}\\n"
    latex += "\\\\begin{tabular}{@{}lcccc@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Model} & \\\\textbf{Accuracy} & \\\\textbf{Precision} & \\\\textbf{F1-Score} & \\\\textbf{ROC-AUC (OVR)} \\\\\\\\ \\\\midrule\\n"
    
    for _, row in df.iterrows():
        latex += f"{row['model'].upper()} & {row['accuracy']:.4f} & {row['precision_macro']:.4f} & {row['f1_macro']:.4f} & {row['roc_auc_ovr_macro']:.4f} \\\\\\\\\\n"
    
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table3_drift_robustness.tex", "w") as f:
        f.write(latex)
    print("Table 3 generated successfully: table3_drift_robustness.tex")

if __name__ == "__main__":
    generate_table()
''',

    "table4_cross_domain.py": '''import pandas as pd

def generate_table(csv_path="cross_domain_clean_model__combined_metrics.csv"):
    """Generates Table 4: Cross-Domain Generalization"""
    try:
        df = pd.read_csv(csv_path)
        df = df[df['task'] == 'supervised']
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{Cross-Domain Generalization (Clean Model to Target Domains)}\\n\\\\label{tab:cross_domain}\\n"
    latex += "\\\\begin{tabular}{@{}llccc@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Target Domain} & \\\\textbf{Model} & \\\\textbf{Accuracy} & \\\\textbf{F1-Score} & \\\\textbf{ROC-AUC} \\\\\\\\ \\\\midrule\\n"
    
    for _, row in df.iterrows():
        latex += f"{row['dataset'].replace('_', '\\\\_')} & {row['model'].upper()} & {row['accuracy']:.4f} & {row['f1_macro']:.4f} & {row['roc_auc_ovr_macro']:.4f} \\\\\\\\\\n"
    
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table4_cross_domain.tex", "w") as f:
        f.write(latex)
    print("Table 4 generated successfully: table4_cross_domain.tex")

if __name__ == "__main__":
    generate_table()
''',

    "table5_anomaly_detection.py": '''import pandas as pd
import os

def generate_table():
    """Generates Table 5: Zero-Shot Anomaly Detection (Autoencoder)"""
    files = [
        "analysis_clean_s__metrics_anomaly.csv", 
        "analysis_unknown_s__metrics_anomaly.csv", 
        "analysis_drift_s__metrics_anomaly.csv",
        "analysis_asym_s__metrics_anomaly.csv"
    ]
    
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{Autoencoder Zero-Shot Anomaly Detection Performance}\\n\\\\label{tab:anomaly_metrics}\\n"
    latex += "\\\\begin{tabular}{@{}lcccc@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Evaluation Domain} & \\\\textbf{ROC-AUC} & \\\\textbf{Avg Precision (PR)} & \\\\textbf{F1-Score} & \\\\textbf{Threshold} \\\\\\\\ \\\\midrule\\n"
    
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            row = df.iloc[0]
            dataset_name = row['dataset'].replace('_', '\\\\_')
            latex += f"{dataset_name} & {row['roc_auc']:.4f} & {row['average_precision']:.4f} & {row['f1_macro']:.4f} & {row['threshold']:.4f} \\\\\\\\\\n"
    
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table5_anomaly_detection.tex", "w") as f:
        f.write(latex)
    print("Table 5 generated successfully: table5_anomaly_detection.tex")

if __name__ == "__main__":
    generate_table()
''',

    "table6_telemetry_features.py": '''def generate_table():
    """Generates Table 6: Physical Telemetry Feature Mapping"""
    latex = "\\\\begin{table}[htbp]\\n\\\\centering\\n\\\\caption{Physical Telemetry Feature Impact Analysis}\\n\\\\label{tab:telemetry_features}\\n"
    latex += "\\\\begin{tabular}{@{}p{4.5cm}p{4cm}@{}}\\n\\\\toprule\\n"
    latex += "\\\\textbf{Telemetry Feature} & \\\\textbf{Primary Vulnerability Indicated} \\\\\\\\ \\\\midrule\\n"
    latex += "Phase Lock Error (rad) & Ornstein-Uhlenbeck Phase Drift \\\\\\\\\\n"
    latex += "Visibility (Interference) & Reference Light Tampering \\\\\\\\\\n"
    latex += "QBER Phase / Bit & Detector Blinding / Wavelength Switching \\\\\\\\\\n"
    latex += "Reference Power (dBm) & Asymmetric Channel Loss \\\\\\\\\\n"
    latex += "Coincidence Rate & Synchronization Jitter \\\\\\\\\\n"
    latex += "\\\\bottomrule\\n\\\\end{tabular}\\n\\\\end{table}\\n"
    
    with open("table6_telemetry_features.tex", "w") as f:
        f.write(latex)
    print("Table 6 generated successfully: table6_telemetry_features.tex")

if __name__ == "__main__":
    generate_table()
'''
}

def build_zip():
    zip_name = "qkd_latex_scripts.zip"
    
    # Write the zip file containing all python scripts
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for script_name, code in table_scripts.items():
            zf.writestr(script_name, code.strip())
            
    print(f"\\nSUCCESS! Bundled {len(table_scripts)} isolated scripts into '{zip_name}'.")
    print("Extract the zip and run individual scripts to dynamically generate your Overleaf-ready .tex tables from the project CSVs.")

if __name__ == "__main__":
    build_zip()
