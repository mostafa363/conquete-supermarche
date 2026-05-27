import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_CSV_PATH, CLEANED_CSV_PATH, MODEL_PATH


def main():
    print("=" * 60)
    print("  SOGOOD — Pipeline complet")
    print("=" * 60)

    # ── Étape 1 : Données ─────────────────────────────────────────
    print("\n[1/5] Collecte des données")
    from src.data_pipeline import fetch_sample_via_api, load_raw_data, clean_data

    if not CLEANED_CSV_PATH.exists():
        if not RAW_CSV_PATH.exists():
            print("  → Récupération via API Open Food Facts...")
            df_raw = fetch_sample_via_api(n_pages=100)
        else:
            df_raw = load_raw_data()

        print("\n[2/5] Nettoyage des données")
        df_clean = clean_data(df_raw)
    else:
        print("  → Données déjà nettoyées, chargement...")
        df_clean = pd.read_csv(CLEANED_CSV_PATH)
        print(f"  → {len(df_clean):,} produits chargés.")

    # ── Étape 2 : Base de données ─────────────────────────────────
    print("\n[3/5] Base de données PostgreSQL")
    try:
        from src.database import init_db, insert_products
        init_db()
        insert_products(df_clean)
    except Exception as e:
        print(f"  [WARN] PostgreSQL ignoré : {e}")
        print("  → Vérifie que PostgreSQL est démarré et le fichier .env configuré.")

    # ── Étape 3 : Analyse ─────────────────────────────────────────
    print("\n[4/5] Analyse exploratoire")
    from src.analysis import describe_dataset
    stats = describe_dataset(df_clean)
    print(f"  → Produits       : {stats['n_products']:,}")
    print(f"  → Marques        : {stats['n_brands']:,}")
    print(f"  → Nutri-Score    : {stats['nutriscore_dist']}")
    print(f"  → Additifs moy.  : {stats['avg_additives']}")

    # ── Étape 4 : Modèle ML ───────────────────────────────────────
    print("\n[5/5] Entraînement du modèle ML")
    if not MODEL_PATH.exists():
        from src.model import train_and_save_model
        pipeline, metrics, _ = train_and_save_model(df_clean)
        print(f"  → Accuracy : {metrics['accuracy']*100:.1f}%")
        print(f"  → F1-score : {metrics['f1_score']:.4f}")
    else:
        print(f"  → Modèle déjà entraîné : {MODEL_PATH}")

    # ── Fin ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Pipeline terminé !")
    print("  Lance l'app avec : streamlit run app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
