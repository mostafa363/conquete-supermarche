import duckdb
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_CSV_PATH, DATA_DIR

DUCK_DB_PATH = DATA_DIR / "sogood.duckdb"


def get_connection():
    return duckdb.connect(str(DUCK_DB_PATH))


def init_duckdb(df: pd.DataFrame):
    con = get_connection()
    con.execute("DROP TABLE IF EXISTS products")
    con.execute("""
        CREATE TABLE products AS SELECT * FROM df
    """)
    count = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    con.close()
    print(f"[DuckDB] {count:,} produits chargés dans sogood.duckdb")
    return count


def query_products(
    search_term=None,
    category=None,
    nutriscore=None,
    allergen_exclude=None,
    limit=50,
) -> pd.DataFrame:
    con = get_connection()
    conditions = ["1=1"]
    if search_term:
        conditions.append(f"LOWER(product_name) LIKE '%{search_term.lower()}%'")
    if category:
        conditions.append(f"LOWER(categories_en) LIKE '%{category.lower()}%'")
    if nutriscore:
        conditions.append(f"nutriscore_grade = '{nutriscore.lower()}'")
    if allergen_exclude:
        for allergen in allergen_exclude:
            conditions.append(
                f"(ingredients_text IS NULL OR LOWER(ingredients_text) NOT LIKE '%{allergen.lower()}%')"
            )
    where = " AND ".join(conditions)
    sql = f"SELECT * FROM products WHERE {where} LIMIT {limit}"
    try:
        result = con.execute(sql).df()
    except Exception:
        result = pd.DataFrame()
    con.close()
    return result


def search_by_barcode(code: str) -> pd.DataFrame:
    con = get_connection()
    try:
        result = con.execute(
            "SELECT * FROM products WHERE code = ? LIMIT 1", [str(code)]
        ).df()
    except Exception:
        result = pd.DataFrame()
    con.close()
    return result


def get_substitutes(row: pd.Series, n=3) -> pd.DataFrame:
    """Return n healthier products in the same category."""
    current_grade = str(row.get("nutriscore_grade", "e")).lower()
    grade_order = ["a", "b", "c", "d", "e"]
    current_idx = grade_order.index(current_grade) if current_grade in grade_order else 4
    better_grades = grade_order[:current_idx]
    if not better_grades:
        return pd.DataFrame()

    category_raw = str(row.get("categories_en", ""))
    first_cat = category_raw.split(",")[0].strip() if category_raw else ""

    con = get_connection()
    grades_list = ", ".join([f"'{g}'" for g in better_grades])
    try:
        if first_cat:
            result = con.execute(f"""
                SELECT * FROM products
                WHERE nutriscore_grade IN ({grades_list})
                  AND LOWER(categories_en) LIKE '%{first_cat.lower()}%'
                  AND product_name != '{str(row.get("product_name","")).replace("'","''")}'
                ORDER BY nutriscore_grade ASC, energy_100g ASC
                LIMIT {n}
            """).df()
        else:
            result = pd.DataFrame()
        if result.empty:
            result = con.execute(f"""
                SELECT * FROM products
                WHERE nutriscore_grade IN ({grades_list})
                ORDER BY nutriscore_grade ASC, energy_100g ASC
                LIMIT {n}
            """).df()
    except Exception:
        result = pd.DataFrame()
    con.close()
    return result


def run_sql(sql: str) -> pd.DataFrame:
    con = get_connection()
    try:
        result = con.execute(sql).df()
    except Exception as e:
        result = pd.DataFrame({"error": [str(e)]})
    con.close()
    return result
