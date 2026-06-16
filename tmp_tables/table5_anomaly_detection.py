import pandas as pd
import os

def generate_table():
    """Generates Table 5: Zero-Shot Anomaly Detection (Autoencoder)"""
    files = [
        "analysis_clean_s__metrics_anomaly.csv", 
        "analysis_unknown_s__metrics_anomaly.csv", 
        "analysis_drift_s__metrics_anomaly.csv",
        "analysis_asym_s__metrics_anomaly.csv"
    ]
    
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Autoencoder Zero-Shot Anomaly Detection Performance}\n\\label{tab:anomaly_metrics}\n"
    latex += "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    latex += "\\textbf{Evaluation Domain} & \\textbf{ROC-AUC} & \\textbf{Avg Precision (PR)} & \\textbf{F1-Score} & \\textbf{Threshold} \\\\ \\midrule\n"
    
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            row = df.iloc[0]
            dataset_name = row['dataset'].replace('_', '\\_')
            latex += f"{dataset_name} & {row['roc_auc']:.4f} & {row['average_precision']:.4f} & {row['f1_macro']:.4f} & {row['threshold']:.4f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table5_anomaly_detection.tex", "w") as f:
        f.write(latex)
    print("Table 5 generated successfully: table5_anomaly_detection.tex")

if __name__ == "__main__":
    generate_table()