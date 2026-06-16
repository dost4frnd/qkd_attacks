import pandas as pd

def generate_table():
    """Generates Table 1: TF-QKD Dataset Variations and Attack Distributions"""
    data = [
        {"Dataset": "tfqkd_clean", "Samples": 6400, "Noise Profile": "Standard", "Asym Loss (dB)": 0, "Phase Drift": "None"},
        {"Dataset": "tfqkd_drift", "Samples": 6400, "Noise Profile": "Ornstein-Uhlenbeck", "Asym Loss (dB)": 0, "Phase Drift": "High"},
        {"Dataset": "tfqkd_asym", "Samples": 6400, "Noise Profile": "Standard", "Asym Loss (dB)": 3, "Phase Drift": "None"},
        {"Dataset": "tfqkd_unknown", "Samples": 7200, "Noise Profile": "Mixed", "Asym Loss (dB)": "0-3", "Phase Drift": "Variable"}
    ]
    df = pd.DataFrame(data)
    
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{TF-QKD Dataset Variations and Attack Distributions}\n\\label{tab:dataset_params}\n"
    latex += "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    latex += "\\textbf{Dataset Variant} & \\textbf{Total Samples} & \\textbf{Phase Drift Noise} & \\textbf{Asymmetric Loss} \\\\ \\midrule\n"
    
    for _, row in df.iterrows():
        latex += f"{row['Dataset'].replace('_', '\\_')} & {row['Samples']} & {row['Noise Profile']} & {row['Asym Loss (dB)']} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table1_dataset_parameters.tex", "w") as f:
        f.write(latex)
    print("Table 1 generated successfully: table1_dataset_parameters.tex")

if __name__ == "__main__":
    generate_table()