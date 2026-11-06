"""
01_load_inspect.py
Purpose: Load the raw dataset and print a full inspection report.
Output : console report only (no files saved yet).
"""
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/soil_weather_crop_disease.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nCOLUMNS:\n", df.columns.tolist())
    print("\nDTYPES:\n", df.dtypes)
    print("\nMISSING VALUES:\n", df.isnull().sum())
    print("\nDUPLICATES:", df.duplicated().sum())
    print("\nFIRST 5 ROWS:\n", df.head())
    print("\nTARGET CANDIDATES (object/category columns):")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        print(f"  {col}: {df[col].nunique()} unique -> {df[col].unique()[:5]}")

if __name__ == "__main__":
    main()