"""
Charge et intègre le dataset Open Food Facts depuis Kaggle.

Dataset : https://www.kaggle.com/datasets/openfoodfacts/world.openfoodfacts.org.products
Fichier  : en.openfoodfacts.org.products.csv  (~2,8 M produits, ~1,5 GB)

Instructions :
    1. pip install kaggle
    2. Placer kaggle.json dans ~/.kaggle/  (API token Kaggle)
    3. python src/kaggle_loader.py
       → télécharge, filtre, nettoie, sauvegarde dans data/

Ou manuellement :
    1. Télécharger depuis Kaggle (lien ci-dessus)
    2. Placer le CSV dans data/en.openfoodfacts.org.products.csv
    3. python src/kaggle_loader.py --local
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, USEFUL_COLUMNS, NUTRISCORE_LABELS, CLEANED_CSV_PATH

KAGGLE_CSV = DATA_DIR / "en.openfoodfacts.org.products.csv"

KAGGLE_COL_MAP = {
    "product_name":        "product_name",
    "brands":              "brands",
    "categories_en":       "categories_en",
    "countries_en":        "countries_en",
    "ingredients_text":    "ingredients_text",
    "nutriscore_score":    "nutriscore_score",
    "nutriscore_grade":    "nutriscore_grade",
    "nova_group":          "nova_group",
    "energy-kcal_100g":    "energy_100g",
    "energy_100g":         "energy_100g",
    "fat_100g":            "fat_100g",
    "saturated-fat_100g":  "saturated-fat_100g",
    "carbohydrates_100g":  "carbohydrates_100g",
    "sugars_100g":         "sugars_100g",
    "fiber_100g":          "fiber_100g",
    "proteins_100g":       "proteins_100g",
    "salt_100g":           "salt_100g",
    "additives_n":         "additives_n",
    "image_url":           "image_url",
    "code":                "code",
}

FEATURE_COLS = [
    "energy_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g", "additives_n",
]


def download_kaggle():
    try:
        import kaggle
        kaggle.api.authenticate()
        print("[Kaggle] Téléchargement du dataset Open Food Facts...")
        kaggle.api.dataset_download_files(
            "openfoodfacts/world.openfoodfacts.org.products",
            path=str(DATA_DIR),
            unzip=True,
        )
        print(f"[Kaggle] Fichier téléchargé dans {DATA_DIR}")
    except Exception as e:
        print(f"[ERREUR] Impossible de télécharger via Kaggle API : {e}")
        print("  → Télécharge manuellement depuis :")
        print("    https://www.kaggle.com/datasets/openfoodfacts/world.openfoodfacts.org.products")
        print(f"  → Place le CSV dans : {KAGGLE_CSV}")
        sys.exit(1)


def load_and_clean(max_products: int = 50_000) -> pd.DataFrame:
    if not KAGGLE_CSV.exists():
        print(f"[ERREUR] Fichier introuvable : {KAGGLE_CSV}")
        print("  Lance d'abord : python src/kaggle_loader.py --download")
        sys.exit(1)

    print(f"[INFO] Lecture du CSV Kaggle ({KAGGLE_CSV.stat().st_size/1e6:.0f} MB)...")

    # Read in chunks to avoid OOM on large file
    chunks = []
    total = 0
    chunk_size = 100_000
    for chunk in pd.read_csv(
        KAGGLE_CSV,
        sep="\t",
        low_memory=False,
        on_bad_lines="skip",
        usecols=lambda c: c in KAGGLE_COL_MAP,
        chunksize=chunk_size,
        encoding="utf-8",
    ):
        # Rename columns
        chunk = chunk.rename(columns={k: v for k, v in KAGGLE_COL_MAP.items() if k in chunk.columns})

        # Keep only European / French products for relevance
        if "countries_en" in chunk.columns:
            chunk = chunk[
                chunk["countries_en"].str.lower().str.contains("france|belgium|switzerland|canada", na=False)
            ]

        # Must have nutriscore and product name
        chunk = chunk.dropna(subset=["nutriscore_grade", "product_name"])
        chunk = chunk[chunk["nutriscore_grade"].str.lower().isin(["a","b","c","d","e"])]

        chunks.append(chunk)
        total += len(chunk)
        print(f"\r[INFO] {total:,} produits filtrés...", end="")

        if total >= max_products:
            break

    print()
    df = pd.concat(chunks, ignore_index=True).head(max_products)
    print(f"[INFO] {len(df):,} produits chargés avant nettoyage.")

    # Normalize nutriscore grade
    df["nutriscore_grade"] = df["nutriscore_grade"].str.lower().str.strip()

    # Numeric columns
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric with median per nutriscore group
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df.groupby("nutriscore_grade")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # Additives
    if "additives_n" in df.columns:
        df["additives_n"] = pd.to_numeric(df["additives_n"], errors="coerce").fillna(0).astype(int)

    # Add label
    df["nutriscore_label"] = df["nutriscore_grade"].map(NUTRISCORE_LABELS)
    df = df.dropna(subset=["nutriscore_label"])
    df["nutriscore_label"] = df["nutriscore_label"].astype(int)

    # Drop duplicates
    df = df.drop_duplicates(subset=["product_name", "brands"]).reset_index(drop=True)

    print(f"[INFO] {len(df):,} produits après nettoyage.")
    return df


def merge_with_existing(df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge Kaggle data with existing API data, drop duplicates."""
    if CLEANED_CSV_PATH.exists():
        df_old = pd.read_csv(CLEANED_CSV_PATH)
        print(f"[INFO] Fusion avec {len(df_old):,} produits existants...")
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["product_name", "brands"]).reset_index(drop=True)
        print(f"[INFO] Total après fusion : {len(df_merged):,} produits")
        return df_merged
    return df_new


def main():
    parser = argparse.ArgumentParser(description="Charge le dataset Kaggle Open Food Facts")
    parser.add_argument("--download", action="store_true", help="Télécharger via Kaggle API")
    parser.add_argument("--local",    action="store_true", help="Fichier CSV déjà présent localement")
    parser.add_argument("--max",      type=int, default=50_000, help="Nombre max de produits (défaut: 50 000)")
    parser.add_argument("--no-merge", action="store_true", help="Remplacer les données existantes sans fusionner")
    args = parser.parse_args()

    if args.download:
        download_kaggle()

    df = load_and_clean(max_products=args.max)

    if not args.no_merge:
        df = merge_with_existing(df)

    out = CLEANED_CSV_PATH
    df.to_csv(out, index=False)
    print(f"\n[OK] Dataset sauvegardé : {out}")
    print(f"     {len(df):,} produits | Nutri-Score : {df['nutriscore_grade'].value_counts().to_dict()}")
    print("\nRelance maintenant le pipeline :")
    print("  python main.py")
    print("  (les modèles seront ré-entraînés sur le nouveau dataset)")


if __name__ == "__main__":
    main()
