import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_CSV_PATH, CLEANED_CSV_PATH, MODEL_PATH, MODEL_DIR

NLP_MODEL_PATH = MODEL_DIR / "nlp_nutriscore_clf.joblib"


def main():
    print("=" * 60)
    print("  SOGOOD — Pipeline complet")
    print("=" * 60)

    # ── Étape 1 : Données ─────────────────────────────────────────
    print("\n[1/6] Collecte des données")
    from src.data_pipeline import fetch_sample_via_api, load_raw_data, clean_data

    if not CLEANED_CSV_PATH.exists():
        if not RAW_CSV_PATH.exists():
            print("  → Récupération via API Open Food Facts...")
            df_raw = fetch_sample_via_api(n_pages=100)
        else:
            df_raw = load_raw_data()

        print("\n[2/6] Nettoyage des données")
        df_clean = clean_data(df_raw)
    else:
        print("  → Données déjà nettoyées, chargement...")
        df_clean = pd.read_csv(CLEANED_CSV_PATH)
        print(f"  → {len(df_clean):,} produits chargés.")

    # ── Étape 2 : Base de données ─────────────────────────────────
    print("\n[3/6] Base de données PostgreSQL")
    try:
        from src.database import init_db, insert_products
        init_db()
        insert_products(df_clean)
    except Exception as e:
        print(f"  [WARN] PostgreSQL ignoré : {e}")

    # ── Étape 3 : Analyse ─────────────────────────────────────────
    print("\n[4/6] Analyse exploratoire")
    from src.analysis import describe_dataset
    stats = describe_dataset(df_clean)
    print(f"  → Produits       : {stats['n_products']:,}")
    print(f"  → Marques        : {stats['n_brands']:,}")
    print(f"  → Nutri-Score    : {stats['nutriscore_dist']}")
    print(f"  → Additifs moy.  : {stats['avg_additives']}")

    # ── Étape 4 : Modèle XGBoost + MLflow ────────────────────────
    print("\n[5/6] Entraînement XGBoost (MLflow tracking)")
    if not MODEL_PATH.exists():
        import mlflow
        mlflow.set_experiment("sogood-nutriscore")
        with mlflow.start_run(run_name="xgboost_baseline"):
            from src.model import train_and_save_model
            pipeline, metrics, _ = train_and_save_model(df_clean)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_samples", stats["n_products"])
            print(f"  → Accuracy : {metrics['accuracy']*100:.1f}% | F1 : {metrics['f1_score']:.4f}")
            print("  → Run MLflow loggé.")
    else:
        print(f"  → Modèle déjà entraîné : {MODEL_PATH}")

    # ── Étape 5 : Modèle NLP Hugging Face ────────────────────────
    print("\n[6/6] Entraînement NLP (Hugging Face sentence-transformers)")
    if not NLP_MODEL_PATH.exists():
        import mlflow
        mlflow.set_experiment("sogood-nutriscore")
        with mlflow.start_run(run_name="nlp_sentence_transformers"):
            from src.nlp_model import train_nlp_model
            metrics_nlp, _, _ = train_nlp_model(df_clean)
            mlflow.log_metric("accuracy", metrics_nlp["accuracy"])
            mlflow.log_metric("f1_score", metrics_nlp["f1_score"])
            mlflow.log_param("model_type", "SentenceTransformer+LR")
            mlflow.log_param("hf_model", metrics_nlp["hf_model"])
            print(f"  → Accuracy NLP : {metrics_nlp['accuracy']*100:.1f}%")
            print("  → Run MLflow loggé.")
    else:
        print(f"  → Modèle NLP déjà entraîné : {NLP_MODEL_PATH}")

    # ── Fin ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Pipeline terminé !")
    print("  App       : streamlit run app/app.py")
    print("  MLflow UI : mlflow ui  (http://localhost:5000)")
    print("=" * 60)


if __name__ == "__main__":
    main()
