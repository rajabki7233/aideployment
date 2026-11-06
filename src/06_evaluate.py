"""
06_evaluate.py
Purpose: Confusion matrix + classification report for the best model.
Input  : models/*.pkl, data/processed/test_fe.csv
Output : outputs/confusion_matrix.png
"""
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

TEST_PATH  = "data/processed/test_fe.csv"
MODEL_PATH = "models/disease_model.pkl"
SCALER_PATH= "models/scaler.pkl"
OUTPUT_DIR = "outputs"
TARGET     = "label"

def main():
    test = pd.read_csv(TEST_PATH)
    X_test = test.drop(columns=[TARGET]); y_test = test[TARGET]

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    preds  = model.predict(scaler.transform(X_test))

    print(classification_report(y_test, preds))

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(y_test, preds), cmap="Blues")
    plt.title("Confusion matrix — best model")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved", f"{OUTPUT_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    main()