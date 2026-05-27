import pandas as pd
import numpy as np
import json
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLUMNS, NUTRISCORE_LABELS, MODEL_PATH, METRICS_PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb


def prepare_ml_data(df: pd.DataFrame):
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[cols].values
    y = df["nutriscore_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train : {len(X_train):,} | Test : {len(X_test):,}")
    return X_train, X_test, y_train, y_test, cols


def train_xgboost(X_train, y_train) -> Pipeline:
    print("[INFO] Entraînement XGBoost...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        ))
    ])
    pipeline.fit(X_train, y_train)
    print("[OK] XGBoost entraîné.")
    return pipeline


def evaluate_model(pipeline, X_test, y_test, name="XGBoost") -> dict:
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"\nAccuracy : {acc*100:.1f}% | F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["A","B","C","D","E"]))
    metrics = {"model": name, "accuracy": round(acc,4), "f1_score": round(f1,4)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def save_model(pipeline):
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[OK] Modèle sauvegardé : {MODEL_PATH}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modèle introuvable. Lance d'abord main.py")
    return joblib.load(MODEL_PATH)


def predict_nutriscore(pipeline, product_values: dict) -> str:
    values = [product_values.get(col, 0) for col in FEATURE_COLUMNS]
    X = np.array(values).reshape(1, -1)
    label = pipeline.predict(X)[0]
    grade_map = {v: k for k, v in NUTRISCORE_LABELS.items()}
    return grade_map[label]


def train_and_save_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test, cols = prepare_ml_data(df)
    pipeline = train_xgboost(X_train, y_train)
    metrics  = evaluate_model(pipeline, X_test, y_test)
    save_model(pipeline)
    return pipeline, metrics, cols
