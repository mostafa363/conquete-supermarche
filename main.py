"""
Orchestrateur principal — pipeline SoGood.

Coordonne les 4 couches indépendantes :
  1. OpenFoodFactsFetcher  → télécharge les données brutes
  2. DataCleaner           → nettoie et transforme
  3. ProductRepository     → persiste dans DuckDB
  4. app/app.py            → présentation (lancée séparément)
"""

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

    # ── Couche 1 : Collecte (Fetcher) ─────────────────────────────
    print("\n[1/6] Collecte des données — OpenFoodFactsFetcher")
    from src.fetcher import OpenFoodFactsFetcher
    fetcher = OpenFoodFactsFetcher()

    if CLEANED_CSV_PATH.exists():
        print("  → Données déjà nettoyées, chargement direct.")
        df_clean = pd.read_csv(CLEANED_CSV_PATH)
        print(f"  → {len(df_clean):,} produits chargés.")
    else:
        if RAW_CSV_PATH.exists():
            print("  → Chargement du CSV brut existant.")
            df_raw = fetcher.load_raw()
        else:
            print("  → Récupération via API Open Food Facts...")
            df_raw = fetcher.fetch_pages(n_pages=100)
            fetcher.save_raw(df_raw)

        # ── Couche 2 : Nettoyage (Cleaner) ────────────────────────
        print("\n[2/6] Nettoyage — DataCleaner")
        from src.cleaner import DataCleaner
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)
        cleaner.save(df_clean)

    # ── Couche 3 : Persistance (Repository) ───────────────────────
    print("\n[3/6] Base de données — ProductRepository (DuckDB)")
    from src.repository import ProductRepository
    repo = ProductRepository()
    repo.save(df_clean)

    # ── Analyse exploratoire ───────────────────────────────────────
    print("\n[4/6] Analyse exploratoire")
    from src.analysis import describe_dataset
    stats = describe_dataset(df_clean)
    print(f"  → Produits       : {stats['n_products']:,}")
    print(f"  → Marques        : {stats['n_brands']:,}")
    print(f"  → Nutri-Score    : {stats['nutriscore_dist']}")
    print(f"  → Additifs moy.  : {stats['avg_additives']:.2f}")

    # ── Entraînement XGBoost ───────────────────────────────────────
    print("\n[5/6] Entraînement XGBoost (MLflow tracking)")
    if MODEL_PATH.exists():
        print(f"  → Modèle déjà entraîné : {MODEL_PATH.name}")
    else:
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

    # ── Entraînement NLP ───────────────────────────────────────────
    print("\n[6/6] Entraînement NLP (Hugging Face sentence-transformers)")
    if NLP_MODEL_PATH.exists():
        print(f"  → Modèle NLP déjà entraîné : {NLP_MODEL_PATH.name}")
    else:
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

    # ── Fin ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OK Pipeline termine !")
    print("  App       : streamlit run app/app.py")
    print("  MLflow UI : mlflow ui  (http://localhost:5000)")
    print("=" * 60)


if __name__ == "__main__":
    main()
