import pandas as pd

def generate_table(csv_path="analysis_drift_s__metrics_supervised.csv"):
    """Generates Table 3: Robustness to Environmental Phase Drift"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Model Robustness under Environmental Phase Drift (DRIFT)}\n\\label{tab:drift_metrics}\n"
    latex += "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{F1-Score} & \\textbf{ROC-AUC (OVR)} \\\\ \\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['model'].upper()} & {row['accuracy']:.4f} & {row['precision_macro']:.4f} & {row['f1_macro']:.4f} & {row['roc_auc_ovr_macro']:.4f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table3_drift_robustness.tex", "w") as f:
        f.write(latex)
    print("Table 3 generated successfully: table3_drift_robustness.tex")

if __name__ == "__main__":
    generate_table()