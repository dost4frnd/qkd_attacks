import pandas as pd

def generate_table(csv_path="cross_domain_clean_model__combined_metrics.csv"):
    """Generates Table 4: Cross-Domain Generalization"""
    try:
        df = pd.read_csv(csv_path)
        df = df[df['task'] == 'supervised']
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
        
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Cross-Domain Generalization (Clean Model to Target Domains)}\n\\label{tab:cross_domain}\n"
    latex += "\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
    latex += "\\textbf{Target Domain} & \\textbf{Model} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{ROC-AUC} \\\\ \\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['dataset'].replace('_', '\\_')} & {row['model'].upper()} & {row['accuracy']:.4f} & {row['f1_macro']:.4f} & {row['roc_auc_ovr_macro']:.4f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table4_cross_domain.tex", "w") as f:
        f.write(latex)
    print("Table 4 generated successfully: table4_cross_domain.tex")

if __name__ == "__main__":
    generate_table()