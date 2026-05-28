"""
Couche streaming — Kafka Producer.

Lit le CSV nettoyé par chunks et publie chaque produit
dans le topic Kafka 'off-products' pour ingestion en temps réel.

Usage :
    python src/kafka_producer.py
    python src/kafka_producer.py --batch 200 --delay 0.2
"""

import argparse
import json
import time
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_CSV_PATH

KAFKA_BROKER = "localhost:9092"
TOPIC        = "off-products"


def create_producer(broker: str):
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"),
        acks="all",
        retries=3,
        linger_ms=10,
    )


def publish(csv_path: Path = CLEANED_CSV_PATH, batch_size: int = 100, delay: float = 0.3):
    print("=" * 55)
    print("  SOGOOD — Kafka Producer")
    print("=" * 55)
    print(f"  Broker : {KAFKA_BROKER}")
    print(f"  Topic  : {TOPIC}")
    print(f"  Source : {csv_path.name}\n")

    if not csv_path.exists():
        print(f"[ERREUR] CSV introuvable : {csv_path}")
        print("  → Lance d'abord : python main.py")
        return

    print("[Producer] Connexion au broker Kafka...")
    try:
        producer = create_producer(KAFKA_BROKER)
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter à Kafka : {e}")
        print("  → Vérifie que Docker tourne : docker-compose up -d kafka")
        return

    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"[Producer] {total:,} produits à publier → topic '{TOPIC}'")
    print("[Producer] Publication en cours... (Ctrl+C pour arrêter)\n")

    sent = 0
    start = time.time()

    for i in range(0, total, batch_size):
        batch = df.iloc[i : i + batch_size]
        for _, row in batch.iterrows():
            record = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            producer.send(TOPIC, value=record)
            sent += 1

        producer.flush()
        elapsed = time.time() - start
        rate = sent / elapsed if elapsed > 0 else 0
        print(f"\r[Producer] {sent:,}/{total:,} produits  |  {rate:.0f} msg/s", end="")
        time.sleep(delay)

    producer.close()
    print(f"\n\n[Producer] ✓ Terminé — {sent:,} messages publiés en {time.time()-start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoGood Kafka Producer")
    parser.add_argument("--batch", type=int, default=100, help="Taille des batches (défaut: 100)")
    parser.add_argument("--delay", type=float, default=0.3, help="Délai entre batches en secondes (défaut: 0.3)")
    args = parser.parse_args()
    publish(batch_size=args.batch, delay=args.delay)
