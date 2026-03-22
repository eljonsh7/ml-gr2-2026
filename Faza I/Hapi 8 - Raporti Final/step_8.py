import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Path configurations
STEP_1_CSV = Path(__file__).resolve().parent.parent / "Hapi 1 - Ngarkimi dhe Bashkimi" / "merged_daily_dataset.csv"
STEP_3_CSV = Path(__file__).resolve().parent.parent / "Hapi 3 - Diskretizimi" / "class_distribution.csv"
STEP_4_CSV = Path(__file__).resolve().parent.parent / "Hapi 5 - Balancimi dhe Mostrimi" / "class_distribution_after_balancing.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

def plot_null_heatmap(df, output_path):
    sample = df.sample(n=min(len(df), 500), random_state=42) if len(df) > 500 else df
    plt.figure(figsize=(12, 6))
    sns.heatmap(sample.isna(), cbar=False, yticklabels=False, cmap="mako")
    plt.title("Missing Values Heatmap (sampled rows)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

def plot_class_balance(before_csv, after_csv, output_path):
    if not before_csv.exists() or not after_csv.exists(): return
    before = pd.read_csv(before_csv)
    after = pd.read_csv(after_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.barplot(data=before, x="target_quantile_class", y="count", ax=axes[0], palette="crest")
    axes[0].set_title("Class Balance (Original)")
    sns.barplot(data=after, x="target_quantile_class", y="count", ax=axes[1], palette="flare")
    axes[1].set_title("Class Balance (Balanced)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

def build_report_file(output_path):
    report_lines = [
        "# Faza I Data Preparation Report",
        "",
        "## Overview",
        "Pipeline-i i përgatitjes së të dhënave mundëson një kalim të pastër nga të dhënat orare te ato ML-ready.",
        "Karakteristikat e procesit:",
        "",
        "### Hapat Kryesorë:",
        "- **Hapi 4 (Detektimi)**: Identifikon anomalitë për analizë.",
        "- **Hapi 5 (Balancimi & Cleaning)**: Fshin anomalitë dhe përgatit setet Train/Test.",
        "- **Hapi 7 (Finalizimi)**: Prodhon datasetin master të pastruar.",
        "",
        "Datasetet finale gjenden në Hapi 5 dhe Hapi 7.",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

def main():
    print("Generating final reporting assets...")
    
    if STEP_1_CSV.exists():
        df1 = pd.read_csv(STEP_1_CSV)
        plot_null_heatmap(df1, OUTPUT_DIR / "null_heatmap.png")
    
    plot_class_balance(STEP_3_CSV, STEP_4_CSV, OUTPUT_DIR / "class_balance_comparison.png")
    
    report_path = OUTPUT_DIR / "phase1_report.md"
    build_report_file(report_path)
    
    # Simple PDF export
    with PdfPages(OUTPUT_DIR / "phase1_report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.01, 0.99, "Faza I: Data Preparation Completed\nCheck phase1_report.md for details.", va="top", ha="left", family="monospace", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print("Step 8 completed. All reports generated.")

if __name__ == "__main__":
    main()
