from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths
indir = Path("cross_domain_clean_model")
outdir = indir

# Load existing combined metrics
combined = pd.read_csv(indir / "combined_metrics.csv")

# Keep only supervised results
sup = combined[combined["task"] == "supervised"].copy()

# Short labels for target domains
domain_names = {
    "analysis_clean_to_clean":   "Test: Clean",
    "analysis_clean_to_drift":   "Test: Drift",
    "analysis_clean_to_asym":    "Test: Asym",
    "analysis_clean_to_unknown": "Test: Unknown",
}

for col in ["accuracy", "f1_macro", "roc_auc_ovr_macro"]:
    if col not in sup.columns:
        continue

    fig, ax = plt.subplots(figsize=(7, 4.5))

    pivot = sup.pivot_table(
        index="source_dir",
        columns="model",
        values=col,
        aggfunc="mean",
    )

    pivot = pivot.reindex(sorted(pivot.index))

    # Replace long labels
    pivot.index = [
        domain_names.get(x, x)
        for x in pivot.index
    ]

    pivot.plot(kind="bar", ax=ax)

    ax.set_title(f"Cross-domain supervised {col}")
    ax.set_xlabel("Target domain (Train domain: Clean)")
    # fig.text(
    # 0.5, -0.03,
    # "C→C: Clean→Clean; C→D: Clean→Drift; "
    # "C→A: Clean→Asymmetric Loss; C→U: Clean→Unknown",
    # ha="center",
    # fontsize=8,
    # )
    ax.set_ylabel(col.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    outfile = outdir / f"cross_domain_supervised_{col}.pdf"
    fig.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {outfile}")

print("Done.")
