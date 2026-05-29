"""
Couche d'accès aux données — Open Food Facts API.

Responsabilité unique : télécharger les données brutes depuis l'API OFF.
Ne fait aucun nettoyage, aucun stockage, aucune présentation.
"""

import requests
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_CSV_PATH, USEFUL_COLUMNS, OFF_DUMP_CSV_PATH

_USER_AGENT = "SoGood/1.0 (educational project; mostafabouchamma@gmail.com)"
_SEARCH_URL  = "https://world.openfoodfacts.org/cgi/search.pl"
_PRODUCT_URL = "https://world.openfoodfacts.org/api/v0/product/{code}.json"


class OpenFoodFactsFetcher:
    """Télécharge des produits depuis l'API Open Food Facts."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT

    # ── API Search ───────────────────────────────────────────────────────

    def fetch_pages(self, n_pages: int = 100, page_size: int = 20) -> pd.DataFrame:
        """Récupère n_pages × page_size produits via l'API de recherche OFF."""
        all_products = []
        fields = ",".join(USEFUL_COLUMNS)

        for page in range(1, n_pages + 1):
            params = {
                "action":    "process",
                "json":      1,
                "page_size": page_size,
                "page":      page,
                "fields":    fields,
            }
            try:
                resp = self._session.get(_SEARCH_URL, params=params, timeout=15)
                resp.raise_for_status()
                products = resp.json().get("products", [])
                if not products:
                    break
                all_products.extend(products)
                print(f"\r[Fetcher] Page {page}/{n_pages} — {len(all_products):,} produits", end="")
            except requests.RequestException as e:
                print(f"\n[Fetcher][WARN] Page {page} ignorée : {e}")

        print(f"\n[Fetcher] {len(all_products):,} produits récupérés.")
        return pd.DataFrame(all_products)

    def fetch_by_barcode(self, code: str) -> dict:
        """Retourne les données brutes d'un produit via son code EAN (API v0)."""
        code = code.strip()
        url  = _PRODUCT_URL.format(code=code)
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # v0 : status == 1 quand trouvé
            if data.get("status") == 1 and data.get("product"):
                return data["product"]
            print(f"[Fetcher] Barcode {code} : status={data.get('status')} — {data.get('status_verbose','')}")
        except requests.RequestException as e:
            print(f"[Fetcher][WARN] Barcode {code} : {e}")
            raise
        return {}

    # ── Persistance brute ────────────────────────────────────────────────

    def save_raw(self, df: pd.DataFrame) -> None:
        """Sauvegarde le DataFrame brut sur disque (CSV)."""
        RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV_PATH, index=False)
        print(f"[Fetcher] Raw CSV sauvegardé : {RAW_CSV_PATH} ({len(df):,} lignes)")

    def load_full_dump(self, sample_n: int = 300_000) -> pd.DataFrame:
        """Charge un échantillon depuis le dump complet OpenFoodFacts (TSV ~2GB)."""
        if not OFF_DUMP_CSV_PATH.exists():
            raise FileNotFoundError(f"Dump OFF introuvable : {OFF_DUMP_CSV_PATH}")
        print(f"[Fetcher] Chargement dump OFF — {sample_n:,} premières lignes...")
        df = pd.read_csv(
            OFF_DUMP_CSV_PATH,
            sep="\t",
            usecols=lambda c: c in USEFUL_COLUMNS,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
            nrows=sample_n,
        )
        print(f"[Fetcher] {len(df):,} lignes chargées depuis le dump OFF.")
        return df

    def load_raw(self) -> pd.DataFrame:
        """Charge le CSV brut depuis le disque."""
        if not RAW_CSV_PATH.exists():
            raise FileNotFoundError(f"CSV brut introuvable : {RAW_CSV_PATH}")
        df = pd.read_csv(
            RAW_CSV_PATH,
            sep="\t",
            usecols=lambda c: c in USEFUL_COLUMNS,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        )
        print(f"[Fetcher] {len(df):,} lignes chargées depuis {RAW_CSV_PATH.name}")
        return df
