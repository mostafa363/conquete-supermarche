import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Product(Base):
    __tablename__ = "products"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    code                 = Column(String(20), unique=True)
    product_name         = Column(String(500), nullable=False)
    brands               = Column(String(300))
    categories_en        = Column(Text)
    countries_en         = Column(String(500))
    ingredients_text     = Column(Text)
    nutriscore_grade     = Column(String(1))
    nutriscore_score     = Column(Float)
    nutriscore_label     = Column(Integer)
    nova_group           = Column(Integer)
    predicted_nutriscore = Column(String(1))
    energy_100g          = Column(Float)
    fat_100g             = Column(Float)
    saturated_fat_100g   = Column(Float)
    carbohydrates_100g   = Column(Float)
    sugars_100g          = Column(Float)
    fiber_100g           = Column(Float)
    proteins_100g        = Column(Float)
    salt_100g            = Column(Float)
    additives_n          = Column(Integer)
    additives_tags       = Column(Text)
    image_url            = Column(Text)


def get_engine():
    return create_engine(DATABASE_URL, echo=False)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("[OK] Tables PostgreSQL créées.")
    return engine


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def insert_products(df: pd.DataFrame, batch_size: int = 1000):
    engine = get_engine()
    rename_map = {"saturated-fat_100g": "saturated_fat_100g"}
    df_db = df.rename(columns=rename_map)
    table_cols = [c.name for c in Product.__table__.columns if c.name != "id"]
    df_db = df_db[[c for c in table_cols if c in df_db.columns]]
    total = len(df_db)
    for i in range(0, total, batch_size):
        chunk = df_db.iloc[i:i + batch_size]
        chunk.to_sql("products", engine, if_exists="append", index=False, method="multi")
        print(f"\r[INFO] Insertion : {min(i + batch_size, total):,}/{total:,}", end="")
    print(f"\n[OK] {total:,} produits insérés.")


def query_products(search_term=None, category=None, nutriscore=None, limit=50) -> pd.DataFrame:
    engine = get_engine()
    query = "SELECT * FROM products WHERE 1=1"
    params = {}
    if search_term:
        query += " AND LOWER(product_name) LIKE :search"
        params["search"] = f"%{search_term.lower()}%"
    if category:
        query += " AND LOWER(categories_en) LIKE :category"
        params["category"] = f"%{category.lower()}%"
    if nutriscore:
        query += " AND nutriscore_grade = :nutriscore"
        params["nutriscore"] = nutriscore.lower()
    query += f" LIMIT {limit}"
    from sqlalchemy import text
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
