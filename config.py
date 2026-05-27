import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
APP_DIR    = BASE_DIR / "app"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "sogood"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

USEFUL_COLUMNS = [
    "code", "product_name", "brands", "categories_en",
    "countries_en", "ingredients_text",
    "nutriscore_score", "nutriscore_grade",
    "nova_group",
    "energy_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g",
    "additives_n", "additives_tags",
    "image_url",
]

NUTRISCORE_LABELS = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

FEATURE_COLUMNS = [
    "energy_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g",
    "additives_n",
]

RAW_CSV_PATH     = DATA_DIR / "open_food_facts_raw.csv"
CLEANED_CSV_PATH = DATA_DIR / "open_food_facts_clean.csv"
MODEL_PATH       = MODEL_DIR / "nutriscore_model.joblib"
METRICS_PATH     = MODEL_DIR / "metrics.json"
