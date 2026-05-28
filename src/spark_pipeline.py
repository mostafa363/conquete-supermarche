"""
Couche Big Data — Pipeline PySpark.

Lit le dump OpenFoodFacts brut (TSV ~2GB), nettoie et transforme
les données via Spark, puis exporte un Parquet propre.

Usage local (via Docker) :
    docker exec sogood-spark-master spark-submit \
        --master spark://spark-master:7077 \
        /opt/bitnami/spark/work/src/spark_pipeline.py \
        --input /opt/bitnami/spark/work/data/en.openfoodfacts.org.products.csv \
        --output /opt/bitnami/spark/work/data/sogood_spark.parquet

Usage local (si Java installé) :
    python src/spark_pipeline.py --master local[*]
"""

import argparse
import sys
from pathlib import Path

FEATURE_COLS = [
    "energy_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g",
]

USEFUL_COLS = [
    "code", "product_name", "brands", "categories_en",
    "countries_en", "ingredients_text",
    "nutriscore_score", "nutriscore_grade", "nova_group",
    "energy_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g", "additives_n", "additives_tags",
    "image_url",
]

VALID_GRADES = ["a", "b", "c", "d", "e"]


def build_session(master: str, app_name: str = "SoGood-Pipeline"):
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )


def run_pipeline(input_path: str, output_path: str, master: str = "local[*]",
                 sample_rows: int = 300_000):
    from pyspark.sql import functions as F
    from pyspark.sql.types import FloatType, IntegerType

    print("=" * 55)
    print("  SOGOOD — Pipeline PySpark")
    print("=" * 55)
    print(f"  Master  : {master}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}\n")

    spark = build_session(master)
    spark.sparkContext.setLogLevel("WARN")

    # ── Lecture ──────────────────────────────────────────────
    print(f"[Spark] Lecture CSV ({sample_rows:,} lignes max)...")
    df = (
        spark.read
        .option("sep", "\t")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("encoding", "UTF-8")
        .csv(input_path)
    )

    # Limiter pour la démo TP
    df = df.limit(sample_rows)

    # Colonnes disponibles uniquement
    available = [c for c in USEFUL_COLS if c in df.columns]
    df = df.select(available)

    # ── Nettoyage ────────────────────────────────────────────
    print("[Spark] Filtrage Nutri-Score + dédoublonnage...")
    df = df.filter(F.col("nutriscore_grade").isin(VALID_GRADES))
    df = df.filter(F.col("product_name").isNotNull())
    if "code" in df.columns:
        df = df.dropDuplicates(["code"])

    print("[Spark] Conversion colonnes numériques...")
    for col in FEATURE_COLS:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(FloatType()))

    print("[Spark] Écrêtage outliers (percentile 99)...")
    for col in FEATURE_COLS:
        if col in df.columns:
            q99 = df.approxQuantile(col, [0.99], 0.01)[0] or 9999.0
            df = (df
                  .withColumn(col, F.when(F.col(col) > q99, q99).otherwise(F.col(col)))
                  .withColumn(col, F.when(F.col(col) < 0, 0.0).otherwise(F.col(col))))

    print("[Spark] Remplissage valeurs manquantes (médiane)...")
    for col in FEATURE_COLS:
        if col in df.columns:
            med = df.approxQuantile(col, [0.5], 0.01)
            median_val = med[0] if med else 0.0
            df = df.fillna({col: median_val})

    if "additives_n" in df.columns:
        df = df.withColumn("additives_n", F.col("additives_n").cast(IntegerType()))
        df = df.fillna({"additives_n": 0})

    # ── Label numérique Nutri-Score ──────────────────────────
    label_map = F.create_map(
        F.lit("a"), F.lit(0),
        F.lit("b"), F.lit(1),
        F.lit("c"), F.lit(2),
        F.lit("d"), F.lit(3),
        F.lit("e"), F.lit(4),
    )
    df = df.withColumn("nutriscore_label", label_map[F.col("nutriscore_grade")])

    # ── Nettoyage marques ────────────────────────────────────
    if "brands" in df.columns:
        df = df.withColumn(
            "brands",
            F.coalesce(F.trim(F.col("brands")), F.lit("Inconnu"))
        )

    # ── Stats finales ────────────────────────────────────────
    count = df.count()
    print(f"\n[Spark] ✓ {count:,} produits propres après nettoyage")
    print(f"[Spark] Écriture Parquet → {output_path}")

    df.write.mode("overwrite").parquet(output_path)

    # Résumé Nutri-Score
    print("\n[Spark] Distribution Nutri-Score :")
    df.groupBy("nutriscore_grade").count().orderBy("nutriscore_grade").show()

    spark.stop()
    print("[Spark] ✓ Pipeline PySpark terminé.\n")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoGood PySpark Pipeline")
    parser.add_argument("--input",  default="data/en.openfoodfacts.org.products.csv")
    parser.add_argument("--output", default="data/sogood_spark.parquet")
    parser.add_argument("--master", default="local[*]",
                        help="Spark master URL (ex: local[*] ou spark://spark-master:7077)")
    parser.add_argument("--rows",   type=int, default=300_000,
                        help="Nombre de lignes à traiter (défaut: 300000)")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.master, args.rows)
