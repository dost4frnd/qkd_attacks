def generate_table():
    """Generates Table 6: Physical Telemetry Feature Mapping"""
    latex = "\\begin{table}[htbp]\n\\centering\n\\caption{Physical Telemetry Feature Impact Analysis}\n\\label{tab:telemetry_features}\n"
    latex += "\\begin{tabular}{@{}p{4.5cm}p{4cm}@{}}\n\\toprule\n"
    latex += "\\textbf{Telemetry Feature} & \\textbf{Primary Vulnerability Indicated} \\\\ \\midrule\n"
    latex += "Phase Lock Error (rad) & Ornstein-Uhlenbeck Phase Drift \\\\\n"
    latex += "Visibility (Interference) & Reference Light Tampering \\\\\n"
    latex += "QBER Phase / Bit & Detector Blinding / Wavelength Switching \\\\\n"
    latex += "Reference Power (dBm) & Asymmetric Channel Loss \\\\\n"
    latex += "Coincidence Rate & Synchronization Jitter \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open("table6_telemetry_features.tex", "w") as f:
        f.write(latex)
    print("Table 6 generated successfully: table6_telemetry_features.tex")

if __name__ == "__main__":
    generate_table()