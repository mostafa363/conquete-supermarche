"""
Couche de persistance et de recherche — DuckDB.

Responsabilité unique : enregistrer les données nettoyées dans DuckDB
et répondre aux requêtes de l'application (recherche, filtrage, substitution).
Ne télécharge rien, ne nettoie rien, n'affiche rien.
"""

import duckdb
import pandas as pd
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

_DB_PATH = DATA_DIR / "sogood.duckdb"
_GRADE_ORDER = ["a", "b", "c", "d", "e"]


class ProductRepository:
    """CRUD + requêtes métier sur la base DuckDB des produits."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = str(db_path)

    # ── Connexion ────────────────────────────────────────────────────────

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path)

    # ── Écriture ─────────────────────────────────────────────────────────

    def save(self, df: pd.DataFrame) -> int:
        """Remplace la table products par le DataFrame fourni."""
        con = self._connect()
        con.execute("DROP TABLE IF EXISTS products")
        con.execute("CREATE TABLE products AS SELECT * FROM df")
        count = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        con.close()
        print(f"[Repository] {count:,} produits enregistrés dans DuckDB.")
        return count

    # ── Lecture / Recherche ───────────────────────────────────────────────

    def find_all(
        self,
        search_term: Optional[str] = None,
        category: Optional[str] = None,
        nutriscore: Optional[str] = None,
        allergens_exclude: Optional[list] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Recherche filtrée multi-critères."""
        conditions = ["1=1"]

        if search_term:
            safe = search_term.lower().replace("'", "''")
            conditions.append(f"LOWER(product_name) LIKE '%{safe}%'")
        if category:
            safe = category.lower().replace("'", "''")
            conditions.append(f"LOWER(categories_en) LIKE '%{safe}%'")
        if nutriscore:
            conditions.append(f"nutriscore_grade = '{nutriscore.lower()}'")
        if allergens_exclude:
            for allergen in allergens_exclude:
                safe = allergen.lower().replace("'", "''")
                conditions.append(
                    f"(ingredients_text IS NULL OR "
                    f"LOWER(ingredients_text) NOT LIKE '%{safe}%')"
                )

        sql = f"SELECT * FROM products WHERE {' AND '.join(conditions)} LIMIT {limit}"
        return self._run(sql)

    def find_by_barcode(self, code: str) -> pd.DataFrame:
        """Recherche un produit par code EAN (exact)."""
        return self._run(
            "SELECT * FROM products WHERE code = ? LIMIT 1",
            params=[str(code)],
        )

    def find_substitutes(self, product_row: pd.Series, n: int = 3) -> pd.DataFrame:
        """Retourne n produits mieux notés dans la même catégorie."""
        current_grade = str(product_row.get("nutriscore_grade", "e")).lower()
        if current_grade not in _GRADE_ORDER:
            current_grade = "e"
        current_idx = _GRADE_ORDER.index(current_grade)
        better_grades = _GRADE_ORDER[:current_idx]
        if not better_grades:
            return pd.DataFrame()

        grades_sql = ", ".join(f"'{g}'" for g in better_grades)
        first_cat = str(product_row.get("categories_en", "")).split(",")[0].strip()
        safe_name = str(product_row.get("product_name", "")).replace("'", "''")
        safe_cat  = first_cat.lower().replace("'", "''")

        if first_cat:
            result = self._run(f"""
                SELECT * FROM products
                WHERE nutriscore_grade IN ({grades_sql})
                  AND LOWER(categories_en) LIKE '%{safe_cat}%'
                  AND product_name != '{safe_name}'
                ORDER BY nutriscore_grade ASC, energy_100g ASC
                LIMIT {n}
            """)
            if not result.empty:
                return result

        return self._run(f"""
            SELECT * FROM products
            WHERE nutriscore_grade IN ({grades_sql})
            ORDER BY nutriscore_grade ASC, energy_100g ASC
            LIMIT {n}
        """)

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Exécute une requête SQL libre (explorateur DuckDB)."""
        con = self._connect()
        try:
            result = con.execute(sql).df()
        except Exception as e:
            result = pd.DataFrame({"error": [str(e)]})
        finally:
            con.close()
        return result

    def count(self) -> int:
        """Retourne le nombre total de produits en base."""
        con = self._connect()
        try:
            n = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        except Exception:
            n = 0
        finally:
            con.close()
        return n

    def schema(self) -> pd.DataFrame:
        """Retourne le schéma de la table products."""
        return self._run("DESCRIBE products")

    # ── Utilitaire privé ─────────────────────────────────────────────────

    def _run(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        con = self._connect()
        try:
            if params:
                result = con.execute(sql, params).df()
            else:
                result = con.execute(sql).df()
        except Exception:
            result = pd.DataFrame()
        finally:
            con.close()
        return result
