"""
05_train_models.py
"""
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = "data/processed/train_fe.csv"
TEST_PATH  = "data/processed/test_fe.csv"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
TARGET     = "label"  # Our target column name

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the data
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]

    # 2. Scale the features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 3. Encode the labels (For XGBoost)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    # 4. Define the models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss"),
    }

    results = []
    best_name, best_model, best_f1 = None, None, -1

    # 5. Train and evaluate
    for name, model in models.items():
        if name == "XGBoost":
            ytr, yte = y_train_enc, y_test_enc
        else:
            ytr, yte = y_train, y_test

        model.fit(X_train_s, ytr)
        preds = model.predict(X_test_s)
        
        acc = accuracy_score(yte, preds)
        f1  = f1_score(yte, preds, average="macro")
        results.append({"Model": name, "Accuracy": acc, "Macro F1": f1})
        
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    # 6. Save the results
    pd.DataFrame(results).sort_values("Macro F1", ascending=False).to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # 7. Save the models and transformers
    joblib.dump(best_model, f"{MODEL_DIR}/disease_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(list(X_train.columns), f"{MODEL_DIR}/feature_columns.pkl")
    joblib.dump(le,         f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(best_name,  f"{MODEL_DIR}/best_model_name.pkl")

    print("Best model:", best_name, "| Macro F1:", round(best_f1, 4))
    print("Saved successfully to", MODEL_DIR)

if __name__ == "__main__":
    main()