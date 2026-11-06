"""
04_feature_engineering.py
Purpose: Add engineered features to the processed splits.
Input  : data/processed/train.csv, test.csv
Output : data/processed/train_fe.csv, test_fe.csv
"""
import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH  = "data/processed/test.csv"
OUT_TRAIN  = "data/processed/train_fe.csv"
OUT_TEST   = "data/processed/test_fe.csv"
TARGET     = "label"

def add_features(df):
    cols = df.columns
    # temperature-humidity interaction (label-favourable index)
    if {"temperature", "humidity"}.issubset(cols):
        df["temp_humidity_idx"] = df["temperature"] * df["humidity"]
    # rainfall ratio if both present
    if {"rainfall", "humidity"}.issubset(cols):
        df["rain_humidity_ratio"] = df["rainfall"] / (df["humidity"] + 1e-6)
    return df

def main():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    train = add_features(train)
    test  = add_features(test)

    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)
    print("Feature engineering done:", train.shape, test.shape)

if __name__ == "__main__":
    main()