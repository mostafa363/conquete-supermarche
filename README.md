# 🥦 SoGood — Analyse nutritionnelle des supermarchés

Plateforme d'analyse de la qualité nutritionnelle des produits alimentaires.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Puis modifie avec tes credentials PostgreSQL
```

## Lancement

```bash
# Pipeline complet (données + modèle)
python main.py

# Application web
streamlit run app/app.py
```

## Structure

| Dossier | Rôle |
|---|---|
| `src/data_pipeline.py` | Collecte & nettoyage Open Food Facts |
| `src/database.py` | PostgreSQL / SQLAlchemy |
| `src/analysis.py` | Visualisations Plotly |
| `src/model.py` | Modèle XGBoost |
| `app/app.py` | Application Streamlit |

## Technologies

Python · pandas · XGBoost · Streamlit · PostgreSQL · Plotly
