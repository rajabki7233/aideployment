"""
03_eda.py
Purpose: Produce EDA visuals for the report.
Input  : data/processed/train.csv
Output : outputs/eda_class_balance.png, outputs/eda_correlation.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

TRAIN_PATH = "data/processed/train.csv"
OUTPUT_DIR = "outputs"
TARGET     = "label"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)

    # 1. Class balance
    plt.figure(figsize=(10, 4))
    df[TARGET].value_counts().plot(kind="bar")
    plt.title("label class distribution (train)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_class_balance.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap (numeric only)
    num = df.select_dtypes(include="number").drop(columns=[TARGET], errors="ignore")
    plt.figure(figsize=(12, 8))
    sns.heatmap(num.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=150)
    plt.close()

    print("EDA figures saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()