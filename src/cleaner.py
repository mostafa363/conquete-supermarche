"""
Couche de nettoyage des données.

Responsabilité unique : transformer un DataFrame brut en DataFrame propre
et exploitable pour le ML et la présentation.
Ne télécharge rien, ne stocke rien, n'affiche rien.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLUMNS, NUTRISCORE_LABELS, CLEANED_CSV_PATH

_VALID_GRADES = {"a", "b", "c", "d", "e"}


class DataCleaner:
    """Nettoie et transforme un DataFrame Open Food Facts brut."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complet : applique chaque étape dans l'ordre."""
        if df.empty:
            raise ValueError("[Cleaner] DataFrame vide — aucun produit à nettoyer.")
        print(f"[Cleaner] Début du nettoyage ({len(df):,} lignes)...")

        df = self._drop_invalid_rows(df)
        df = self._clean_numeric_columns(df)
        df = self._clip_outliers(df)
        df = self._fill_missing_numeric(df)
        df = self._clean_categorical_columns(df)
        df = self._add_nutriscore_label(df)
        df = df.reset_index(drop=True)

        print(f"[Cleaner] {len(df):,} produits propres.")
        return df

    # ── Étapes internes ──────────────────────────────────────────────────

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les doublons et les lignes sans nom ou sans Nutri-Score valide."""
        n_before = len(df)
        df = df.drop_duplicates(
            subset=["code"] if "code" in df.columns else None
        )
        df = df.dropna(subset=["product_name"])
        df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.lower().str.strip()
        df = df[df["nutriscore_grade"].isin(_VALID_GRADES)]
        print(f"  [Cleaner] Lignes invalides supprimées : {n_before - len(df):,}")
        return df

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les colonnes nutritionnelles en float."""
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Écrête les valeurs aberrantes au percentile 99 et supprime les négatifs."""
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                continue
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=0, upper=q99)
        return df

    def _fill_missing_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remplace les NaN par la médiane du groupe Nutri-Score correspondant."""
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                continue
            df[col] = df.groupby("nutriscore_grade")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())
        if "additives_n" in df.columns:
            df["additives_n"] = pd.to_numeric(df["additives_n"], errors="coerce").fillna(0).astype(int)
        return df

    def _clean_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes textuelles (marques, catégories)."""
        if "brands" in df.columns:
            df["brands"] = df["brands"].fillna("Inconnu").str.strip().str.title()
        else:
            df["brands"] = "Inconnu"

        if "categories_en" in df.columns:
            df["categories_en"] = df["categories_en"].fillna("Uncategorized")
        else:
            df["categories_en"] = "Uncategorized"
        return df

    def _add_nutriscore_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute la colonne entière nutriscore_label (0=A … 4=E) pour le ML."""
        df["nutriscore_label"] = df["nutriscore_grade"].map(NUTRISCORE_LABELS)
        df = df.dropna(subset=["nutriscore_label"])
        df["nutriscore_label"] = df["nutriscore_label"].astype(int)
        return df

    # ── Persistance CSV ──────────────────────────────────────────────────

    def save(self, df: pd.DataFrame) -> None:
        """Sauvegarde le DataFrame nettoyé en CSV."""
        CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_CSV_PATH, index=False)
        print(f"[Cleaner] Sauvegardé : {CLEANED_CSV_PATH} ({len(df):,} produits)")

    def load(self) -> pd.DataFrame:
        """Charge le CSV nettoyé depuis le disque."""
        if not CLEANED_CSV_PATH.exists():
            raise FileNotFoundError(f"CSV nettoyé introuvable : {CLEANED_CSV_PATH}")
        df = pd.read_csv(CLEANED_CSV_PATH)
        print(f"[Cleaner] {len(df):,} produits chargés depuis {CLEANED_CSV_PATH.name}")
        return df
