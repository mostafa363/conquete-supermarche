import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLUMNS


def describe_dataset(df: pd.DataFrame) -> dict:
    return {
        "n_products":      len(df),
        "n_brands":        df["brands"].nunique() if "brands" in df.columns else 0,
        "n_categories":    df["categories_en"].nunique() if "categories_en" in df.columns else 0,
        "nutriscore_dist": df["nutriscore_grade"].value_counts().to_dict(),
        "nova_dist":       df["nova_group"].value_counts().to_dict() if "nova_group" in df.columns else {},
        "avg_additives":   round(df["additives_n"].mean(), 2) if "additives_n" in df.columns else 0,
        "avg_sugar":       round(df["sugars_100g"].mean(), 2) if "sugars_100g" in df.columns else 0,
        "avg_salt":        round(df["salt_100g"].mean(), 2) if "salt_100g" in df.columns else 0,
    }


COLOR_MAP = {"a": "#1fa37b", "b": "#85bb2f", "c": "#ffcc00", "d": "#ff6600", "e": "#ee3333"}


def plot_nutriscore_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["nutriscore_grade"].value_counts().reset_index()
    counts.columns = ["grade", "count"]
    counts = counts.sort_values("grade")
    fig = px.bar(counts, x="grade", y="count", color="grade",
                 color_discrete_map=COLOR_MAP,
                 title="Distribution des Nutri-Scores",
                 labels={"grade": "Nutri-Score", "count": "Produits"},
                 text="count")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white")
    return fig


def plot_top_brands_controverses(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    bad = df[df["nutriscore_grade"].isin(["d", "e"])]
    top = bad["brands"].value_counts().head(top_n).reset_index()
    top.columns = ["brand", "count"]
    fig = px.bar(top, x="count", y="brand", orientation="h",
                 title=f"Top {top_n} marques controversées (Nutri-Score D/E)",
                 color="count", color_continuous_scale="Reds")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu",
                    title="Corrélations entre valeurs nutritionnelles", aspect="auto")
    return fig


def plot_nutriscore_by_category(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    df = df.copy()
    df["main_category"] = df["categories_en"].str.split(",").str[0].str.strip()
    top_cats = df["main_category"].value_counts().head(top_n).index
    subset = df[df["main_category"].isin(top_cats)]
    avg = subset.groupby("main_category")["nutriscore_label"].mean().sort_values().reset_index()
    avg.columns = ["category", "avg_nutriscore"]
    fig = px.bar(avg, x="avg_nutriscore", y="category", orientation="h",
                 title="Score nutritionnel moyen par catégorie",
                 color="avg_nutriscore", color_continuous_scale="RdYlGn_r")
    return fig


def plot_additives_vs_nutriscore(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="nutriscore_grade", y="additives_n",
                 color="nutriscore_grade", color_discrete_map=COLOR_MAP,
                 title="Nombre d'additifs selon le Nutri-Score",
                 category_orders={"nutriscore_grade": ["a", "b", "c", "d", "e"]})
    fig.update_layout(showlegend=False)
    return fig


def plot_sugar_salt_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(5000, len(df)))
    fig = px.scatter(sample, x="sugars_100g", y="salt_100g",
                     color="nutriscore_grade", color_discrete_map=COLOR_MAP,
                     title="Sucres vs Sel par Nutri-Score",
                     opacity=0.6, hover_data=["product_name", "brands"])
    return fig
