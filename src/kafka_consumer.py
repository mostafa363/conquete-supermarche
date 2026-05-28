"""
Couche streaming — Kafka Consumer.

Consomme les messages du topic 'off-products' et les insère
progressivement dans DuckDB par batches.

Usage :
    python src/kafka_consumer.py
"""

import json
import time
import pandas as pd
import duckdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

KAFKA_BROKER     = "localhost:9092"
TOPIC            = "off-products"
GROUP_ID         = "sogood-consumer-group"
BATCH_INSERT     = 500
TIMEOUT_MS       = 15_000

_DB_PATH = str(DATA_DIR / "sogood.duckdb")


def stream_insert(rows: list):
    """Insère un batch de dicts dans DuckDB (upsert sur code)."""
    if not rows:
        return 0
    con = duckdb.connect(_DB_PATH)
    df = pd.DataFrame(rows)
    try:
        # Crée la table si elle n'existe pas encore
        con.execute("CREATE TABLE IF NOT EXISTS products AS SELECT * FROM df WHERE 1=0")
        con.execute("INSERT INTO products SELECT * FROM df")
    except Exception as e:
        # Si schéma incompatible, on remplace
        con.execute("DROP TABLE IF EXISTS products")
        con.execute("CREATE TABLE products AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    con.close()
    return count


def consume():
    from kafka import KafkaConsumer

    print("=" * 55)
    print("  SOGOOD — Kafka Consumer")
    print("=" * 55)
    print(f"  Broker  : {KAFKA_BROKER}")
    print(f"  Topic   : {TOPIC}")
    print(f"  Groupe  : {GROUP_ID}")
    print(f"  DuckDB  : {_DB_PATH}\n")

    print("[Consumer] Connexion au broker Kafka...")
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=TIMEOUT_MS,
        )
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter à Kafka : {e}")
        print("  → Vérifie que Docker tourne : docker-compose up -d kafka")
        return

    buffer = []
    total  = 0
    start  = time.time()

    print(f"[Consumer] En écoute sur '{TOPIC}'... (timeout {TIMEOUT_MS//1000}s sans message)\n")

    try:
        for msg in consumer:
            buffer.append(msg.value)

            if len(buffer) >= BATCH_INSERT:
                db_count = stream_insert(buffer)
                total += len(buffer)
                elapsed = time.time() - start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"[Consumer] {total:,} consommés | {db_count:,} en DuckDB | {rate:.0f} msg/s")
                buffer = []

    except KeyboardInterrupt:
        print("\n[Consumer] Interruption manuelle.")
    finally:
        if buffer:
            db_count = stream_insert(buffer)
            total += len(buffer)
            print(f"[Consumer] Flush final : {total:,} produits | {db_count:,} en DuckDB")
        consumer.close()

    print(f"\n[Consumer] ✓ Terminé — {total:,} messages consommés en {time.time()-start:.1f}s")


if __name__ == "__main__":
    consume()
