import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUTRISCORE_LABELS, MODEL_DIR

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

NLP_MODEL_PATH  = MODEL_DIR / "nlp_nutriscore_clf.joblib"
NLP_METRICS_PATH = MODEL_DIR / "nlp_metrics.json"
SENTENCE_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"


def build_text_corpus(df: pd.DataFrame):
    """Concatène nom + ingrédients pour enrichir le contexte textuel."""
    df = df.copy()
    df["text"] = (
        df["product_name"].fillna("") + " "
        + df.get("ingredients_text", pd.Series([""] * len(df))).fillna("")
        + " "
        + df.get("categories_en", pd.Series([""] * len(df))).fillna("")
    ).str.strip()
    df = df[df["text"].str.len() > 10]
    return df["text"].tolist(), df["nutriscore_label"].values


def train_nlp_model(df: pd.DataFrame):
    """
    Pipeline NLP :
      1. Hugging Face sentence-transformers → embeddings multilingues
      2. Logistic Regression → classification Nutri-Score A‑E
    """
    print("[INFO] Chargement du modèle Hugging Face :", SENTENCE_MODEL)
    embedder = SentenceTransformer(SENTENCE_MODEL)

    X_text, y = build_text_corpus(df)
    print(f"[INFO] Encodage de {len(X_text):,} descriptions produits...")
    X = embedder.encode(X_text, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Entraînement du classificateur (LogisticRegression)...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[NLP] Accuracy : {acc*100:.1f}% | F1 : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["A","B","C","D","E"]))

    metrics = {
        "model":      "SentenceTransformer + LogisticRegression",
        "hf_model":   SENTENCE_MODEL,
        "accuracy":   round(acc, 4),
        "f1_score":   round(f1, 4),
        "n_samples":  len(X_text),
    }
    with open(NLP_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(clf, NLP_MODEL_PATH)
    print(f"[OK] Classificateur NLP sauvegardé : {NLP_MODEL_PATH}")
    return metrics, embedder, clf


def load_nlp_model():
    if not NLP_MODEL_PATH.exists():
        raise FileNotFoundError("Modèle NLP introuvable. Lance d'abord main.py")
    embedder = SentenceTransformer(SENTENCE_MODEL)
    clf = joblib.load(NLP_MODEL_PATH)
    return embedder, clf


def predict_nutriscore_nlp(embedder, clf, text: str) -> tuple[str, dict]:
    """Retourne le grade prédit et les probabilités par classe."""
    emb = embedder.encode([text], convert_to_numpy=True)
    label = clf.predict(emb)[0]
    proba = clf.predict_proba(emb)[0]
    grade_map = {v: k for k, v in NUTRISCORE_LABELS.items()}
    proba_dict = {grade_map[i]: round(float(p), 3) for i, p in enumerate(proba)}
    return grade_map[label], proba_dict
