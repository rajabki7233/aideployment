"""
07_feature_importance.py
Purpose: Plot top-15 feature importances.
Input  : models/*.pkl, models/feature_columns.pkl
Output : outputs/feature_importance.png
"""
import pandas as pd

import joblib
import matplotlib.pyplot as plt

MODEL_PATH   = "models/disease_model.pkl"
COLS_PATH    = "models/feature_columns.pkl"
OUTPUT_DIR   = "outputs"

def main():
    model = joblib.load(MODEL_PATH)
    cols  = joblib.load(COLS_PATH)

    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ (e.g. Logistic Regression). Skipping.")
        return

    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    imp.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 15 feature importances")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print("Saved", f"{OUTPUT_DIR}/feature_importance.png")

if __name__ == "__main__":
    main()