# 🥦 SoGood — Analyse nutritionnelle Big Data

Plateforme complète d'analyse de la qualité nutritionnelle des produits alimentaires, construite sur une architecture **Big Data** : Kafka · PySpark · DuckDB · XGBoost · NLP Hugging Face · Streamlit.

> Projet IPSSI — Open Data / Big Data TP

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SOURCE DE DONNÉES                          │
│   OpenFoodFacts Dump (2.3 GB TSV)  +  API REST (EAN barcode)   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│               COUCHE STREAMING — APACHE KAFKA                   │
│   kafka_producer.py  →  Topic: off-products  →  kafka_consumer  │
│   (Docker · KRaft mode · port 9092)                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│               COUCHE TRAITEMENT — APACHE SPARK                  │
│   spark_pipeline.py : nettoyage · dédoublonnage · Parquet       │
│   (Docker · spark-master:7077 · spark-worker)                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  COUCHE STOCKAGE — DUCKDB                       │
│   Table products (52k produits)  +  users  +  user_substitutes  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              COUCHE MACHINE LEARNING                            │
│   XGBoost (93.6% accuracy)  +  NLP MiniLM-L12-v2 (HuggingFace) │
│   MLflow experiment tracking                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              APPLICATION WEB — STREAMLIT                        │
│   10 pages : Dashboard · Recherche · Analyses · Prédiction IA  │
│              NLP · Substitution · Comparaison · DuckDB SQL      │
│              Architecture Big Data · Auth utilisateurs          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fonctionnalités

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | Hero section, KPI cards, 4 graphiques Plotly |
| 🔍 **Recherche** | Texte + code-barres EAN (API OFF en fallback) |
| 📊 **Analyses** | Distributions, corrélations, feature importance, matrice de confusion |
| 🤖 **Prédiction IA** | XGBoost depuis valeurs nutritionnelles |
| 🧠 **Prédiction NLP** | Hugging Face sentence-transformers depuis texte ingrédients |
| 🔄 **Substitution** | Alternatives plus saines dans la même catégorie |
| ⚖️ **Comparaison** | Side-by-side + radar chart |
| 🗄️ **Explorateur SQL** | DuckDB interactif avec requêtes prédéfinies |
| 🏗️ **Architecture** | Diagramme pipeline + stats en direct |
| 👤 **Auth** | Inscription / connexion, sauvegarde de substituts |

---

## Stack technique

| Couche | Technologies |
|--------|-------------|
| Données | OpenFoodFacts (2.3 GB TSV), API REST |
| Streaming | Apache Kafka (KRaft, Docker), kafka-python |
| Big Data | Apache Spark 3.x (Docker), PySpark |
| Stockage | DuckDB (embarqué, sans serveur) |
| ML | XGBoost, scikit-learn, MLflow |
| NLP | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Frontend | Streamlit, Plotly |
| Langage | Python 3.13 |

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/mostafa363/conquete-supermarche.git
cd conquete-supermarche/sogood

# 2. Environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Dépendances
pip install -r requirements.txt

# 4. Placer le dump OpenFoodFacts (optionnel, ~2.3 GB)
# Télécharger depuis https://world.openfoodfacts.org/data
# Placer dans : data/en.openfoodfacts.org.products.csv
```

---

## Lancement

### 1. Pipeline ML (données + modèles)
```bash
# Windows
$env:PYTHONUTF8="1"; .\venv\Scripts\python.exe main.py

# Linux/Mac
python main.py
```

### 2. Infrastructure Big Data (Docker requis)
```bash
# Démarrer Kafka + Spark
docker-compose up -d

# Produire des messages Kafka
python src/kafka_producer.py

# Consommer et insérer dans DuckDB
python src/kafka_consumer.py

# Pipeline Spark (depuis le container)
docker exec sogood-spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark/work/src/spark_pipeline.py \
  --input /opt/spark/work/data/en.openfoodfacts.org.products.csv \
  --output /opt/spark/work/data/sogood_spark.parquet
```

### 3. Application web
```bash
streamlit run app/app.py
# → http://localhost:8501
```

---

## Structure du projet

```
sogood/
├── app/
│   └── app.py                  # Application Streamlit (10 pages)
├── src/
│   ├── fetcher.py              # API OpenFoodFacts
│   ├── analysis.py             # Visualisations Plotly
│   ├── model.py                # XGBoost pipeline
│   ├── nlp_model.py            # NLP Hugging Face
│   ├── repository.py           # DuckDB CRUD
│   ├── auth.py                 # Authentification utilisateurs
│   ├── kafka_producer.py       # Producteur Kafka
│   ├── kafka_consumer.py       # Consommateur Kafka → DuckDB
│   └── spark_pipeline.py       # Pipeline PySpark
├── notebooks/
│   └── 01_eda.ipynb            # Analyse exploratoire
├── data/
│   ├── sogood.duckdb           # Base DuckDB
│   └── open_food_facts_clean.csv
├── models/
│   ├── nutriscore_model.joblib # XGBoost entraîné
│   ├── nlp_nutriscore_clf.joblib
│   ├── metrics.json            # Accuracy, F1
│   └── nlp_metrics.json
├── docker-compose.yml          # Kafka + Spark
├── config.py                   # Chemins et constantes
├── main.py                     # Entrée pipeline ML
└── requirements.txt
```

---

## Performances des modèles

| Modèle | Accuracy | F1-score | Description |
|--------|----------|----------|-------------|
| **XGBoost** | **93.6%** | 0.936 | 9 features nutritionnelles (énergie, sucres, sel…) |
| **NLP HuggingFace** | ~70% | ~0.70 | Embeddings 384-dim sur nom + ingrédients |

---

## Auteur

**Mostafa Bouchamma** — Projet IPSSI Big Data / Open Data
