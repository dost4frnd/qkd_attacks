import pandas as pd

def generate_table(csv_path="analysis_clean_s__metrics_supervised.csv"):
    """Generates Table 2: Same-Domain Supervised Performance (CLEAN)"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Same-Domain Supervised Performance Metrics (CLEAN)}\n\\label{tab:clean_metrics}\n"
    latex += "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\ \\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['model'].upper()} & {row['accuracy']:.4f} & {row['precision_macro']:.4f} & {row['recall_macro']:.4f} & {row['f1_macro']:.4f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table2_clean_supervised.tex", "w") as f:
        f.write(latex)
    print("Table 2 generated successfully: table2_clean_supervised.tex")

if __name__ == "__main__":
    generate_table()