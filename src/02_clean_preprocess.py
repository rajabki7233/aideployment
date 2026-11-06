"""
02_clean_preprocess.py
Purpose: Clean data, encode categoricals, split train/test.
Input  : data/raw/soil_weather_crop_disease.csv
Output : data/processed/train.csv, data/processed/test.csv
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH   = "data/raw/soil_weather_crop_disease.csv"
OUTPUT_DIR  = "data/processed"
TARGET      = "label"          # <-- RENAME if 01 shows a different target column

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # --- basic cleaning ---
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- separate target ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # --- encode categorical features (one-hot) ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        print(f"One-hot encoded: {cat_cols}")

    # --- fill numeric NaNs with median ---
    X = X.fillna(X.median(numeric_only=True))

    # --- stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train = X_train.copy(); train[TARGET] = y_train
    test  = X_test.copy();  test[TARGET]  = y_test
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("Saved:", train.shape, "->", f"{OUTPUT_DIR}/train.csv")
    print("Saved:", test.shape,  "->", f"{OUTPUT_DIR}/test.csv")

if __name__ == "__main__":
    main()