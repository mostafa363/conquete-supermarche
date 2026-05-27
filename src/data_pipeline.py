import requests
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_CSV_PATH, CLEANED_CSV_PATH,
    USEFUL_COLUMNS, FEATURE_COLUMNS, NUTRISCORE_LABELS
)


def fetch_sample_via_api(n_pages: int = 100) -> pd.DataFrame:
    """Récupère un échantillon via l'API Open Food Facts (~2000 produits)."""
    base_url = "https://world.openfoodfacts.org/cgi/search.pl"
    headers = {"User-Agent": "SoGood/1.0 (educational project; mostafabouchamma@gmail.com)"}
    all_products = []

    for page in range(1, n_pages + 1):
        params = {
            "action":    "process",
            "json":      1,
            "page_size": 20,
            "page":      page,
            "fields":    ",".join(USEFUL_COLUMNS),
        }
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=15)
            data = resp.json()
            products = data.get("products", [])
            if not products:
                break
            all_products.extend(products)
            print(f"\r[INFO] Page {page}/{n_pages} — {len(all_products)} produits", end="")
        except Exception as e:
            print(f"\n[WARN] Erreur page {page} : {e}")

    print(f"\n[OK] {len(all_products)} produits récupérés.")
    df = pd.DataFrame(all_products)
    df.to_csv(RAW_CSV_PATH, index=False)
    return df


def download_full_dataset():
    """Télécharge le CSV complet Open Food Facts (~800 Mo)."""
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    print("[INFO] Téléchargement du dataset complet...")
    response = requests.get(url, stream=True)
    gz_path = str(RAW_CSV_PATH) + ".gz"
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(gz_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total * 100) if total else 0
            print(f"\r[INFO] {pct:.1f}%", end="")
    print("\n[OK] Téléchargement terminé.")


def load_raw_data(sample_size: int = 200_000) -> pd.DataFrame:
    """Charge le CSV brut."""
    print("[INFO] Chargement du dataset brut...")
    df = pd.read_csv(
        RAW_CSV_PATH,
        sep="\t",
        usecols=lambda c: c in USEFUL_COLUMNS,
        nrows=sample_size,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    print(f"[OK] {len(df):,} lignes chargées.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de nettoyage."""
    if df.empty or len(df) == 0:
        raise ValueError("DataFrame vide — aucun produit récupéré. Vérifie ta connexion.")
    print(f"[INFO] Nettoyage ({len(df):,} lignes)...")

    df = df.drop_duplicates(subset=["code"] if "code" in df.columns else None)
    df = df.dropna(subset=["product_name"])
    df = df[df["nutriscore_grade"].isin(["a", "b", "c", "d", "e"])]

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    for col in numeric_cols:
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=0, upper=q99)
        df[col] = df[col].fillna(df[col].median())

    df["nutriscore_label"] = df["nutriscore_grade"].map(NUTRISCORE_LABELS)

    if "additives_n" in df.columns:
        df["additives_n"] = pd.to_numeric(df["additives_n"], errors="coerce").fillna(0)

    if "brands" in df.columns:
        df["brands"] = df["brands"].fillna("Inconnu").str.strip().str.title()
    else:
        df["brands"] = "Inconnu"

    if "categories_en" in df.columns:
        df["categories_en"] = df["categories_en"].fillna("Uncategorized")
    else:
        df["categories_en"] = "Uncategorized"

    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"[OK] {len(df):,} produits propres sauvegardés.")
    return df
