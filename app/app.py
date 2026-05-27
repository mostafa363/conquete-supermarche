import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_CSV_PATH, FEATURE_COLUMNS, NUTRISCORE_LABELS
from src.analysis import (
    plot_nutriscore_distribution, plot_top_brands_controverses,
    plot_correlation_matrix, plot_nutriscore_by_category,
    plot_additives_vs_nutriscore, plot_sugar_salt_scatter,
    describe_dataset,
)
from src.model import load_model, predict_nutriscore

st.set_page_config(
    page_title="SoGood — Qualité nutritionnelle",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-title { font-size:2.4rem; font-weight:700; color:#1D7A5E; margin-bottom:0; }
    .subtitle  { color:#888; margin-top:0; margin-bottom:2rem; }
    .ns-a { color:#1fa37b; font-size:1.8rem; font-weight:700; }
    .ns-b { color:#85bb2f; font-size:1.8rem; font-weight:700; }
    .ns-c { color:#ffcc00; font-size:1.8rem; font-weight:700; }
    .ns-d { color:#ff6600; font-size:1.8rem; font-weight:700; }
    .ns-e { color:#ee3333; font-size:1.8rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not CLEANED_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEANED_CSV_PATH)


@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception:
        return None


df    = load_data()
model = get_model()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥦 SoGood")
    st.caption("Analyse nutritionnelle des supermarchés")
    st.divider()

    page = st.radio("", [
        "🏠 Dashboard",
        "🔍 Recherche produits",
        "📊 Analyses détaillées",
        "🤖 Prédiction IA",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Filtres**")

    ns_filter = st.multiselect(
        "Nutri-Score",
        ["a", "b", "c", "d", "e"],
        default=["a", "b", "c", "d", "e"],
        format_func=str.upper,
    )

    cat_filter = "Toutes"
    if not df.empty and "categories_en" in df.columns:
        cats = ["Toutes"] + sorted(
            df["categories_en"].str.split(",").str[0].str.strip()
            .dropna().unique().tolist()
        )[:80]
        cat_filter = st.selectbox("Catégorie", cats)

# Appliquer filtres
if not df.empty:
    dff = df[df["nutriscore_grade"].isin(ns_filter)].copy()
    if cat_filter != "Toutes":
        dff = dff[dff["categories_en"].str.contains(cat_filter, na=False)]
else:
    dff = df.copy()

# ── PAGE 1 : Dashboard ───────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<p class="big-title">🥦 SoGood</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Plateforme d\'analyse de la qualité nutritionnelle</p>', unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ Aucune donnée. Lance d'abord `python main.py`")
        st.stop()

    s = describe_dataset(dff)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛒 Produits",       f"{s['n_products']:,}")
    c2.metric("🏷️ Marques",        f"{s['n_brands']:,}")
    c3.metric("📂 Catégories",     f"{s['n_categories']:,}")
    c4.metric("⚗️ Additifs moy.",  f"{s['avg_additives']:.1f}")

    st.divider()
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_nutriscore_distribution(dff),    use_container_width=True)
    c2.plotly_chart(plot_additives_vs_nutriscore(dff),    use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_top_brands_controverses(dff),    use_container_width=True)
    c2.plotly_chart(plot_nutriscore_by_category(dff),     use_container_width=True)

# ── PAGE 2 : Recherche ───────────────────────────────────────────────────
elif page == "🔍 Recherche produits":
    st.header("🔍 Recherche de produits")

    c1, c2, c3 = st.columns([3, 2, 1])
    term  = c1.text_input("Nom du produit", placeholder="Ex: Nutella, yaourt, coca...")
    grade = c2.selectbox("Nutri-Score", ["Tous", "A", "B", "C", "D", "E"])
    c3.write("")
    btn   = c3.button("Rechercher", use_container_width=True)

    results = dff.copy()
    if term:
        results = results[results["product_name"].str.lower().str.contains(term.lower(), na=False)]
    if grade != "Tous":
        results = results[results["nutriscore_grade"] == grade.lower()]

    st.info(f"**{len(results):,}** produit(s) trouvé(s)")

    if not results.empty:
        grade_emoji = {"a":"🟢","b":"🟡","c":"🟠","d":"🔴","e":"🔴"}
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.head(30).iterrows()):
            g = str(row.get("nutriscore_grade","?")).lower()
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{str(row['product_name'])[:50]}**")
                    st.caption(f"🏷️ {row.get('brands','Inconnu')}")
                    st.markdown(f"Nutri-Score : {grade_emoji.get(g,'⚪')} **{g.upper()}**")
                    if row.get("sugars_100g"):
                        st.caption(f"🍬 Sucres : {row['sugars_100g']:.1f}g | 🧂 Sel : {row.get('salt_100g',0):.2f}g")
                    if row.get("additives_n"):
                        st.caption(f"⚗️ {int(row['additives_n'])} additifs")

# ── PAGE 3 : Analyses ────────────────────────────────────────────────────
elif page == "📊 Analyses détaillées":
    st.header("📊 Analyses nutritionnelles")

    t1, t2, t3 = st.tabs(["📊 Distributions", "🔗 Corrélations", "🍬 Sucres & Sel"])

    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_nutriscore_distribution(dff),  use_container_width=True)
        c2.plotly_chart(plot_top_brands_controverses(dff),  use_container_width=True)
        st.plotly_chart(plot_nutriscore_by_category(dff),   use_container_width=True)
    with t2:
        st.plotly_chart(plot_correlation_matrix(dff),       use_container_width=True)
    with t3:
        st.plotly_chart(plot_sugar_salt_scatter(dff),       use_container_width=True)

# ── PAGE 4 : Prédiction ─────────────────────────────────────────────────
elif page == "🤖 Prédiction IA":
    st.header("🤖 Prédiction du Nutri-Score par IA")
    st.info("Renseignez les valeurs nutritionnelles (pour 100g) — le modèle XGBoost prédit le Nutri-Score.")

    if model is None:
        st.error("Modèle non disponible. Lance `python main.py` pour l'entraîner.")
        st.stop()

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            energy   = st.number_input("⚡ Énergie (kcal)",       min_value=0.0, value=200.0, step=10.0)
            fat      = st.number_input("🥩 Graisses (g)",          min_value=0.0, value=5.0)
            sat_fat  = st.number_input("🧈 Graisses saturées (g)", min_value=0.0, value=2.0)
            carbs    = st.number_input("🍞 Glucides (g)",          min_value=0.0, value=30.0)
        with c2:
            sugars   = st.number_input("🍬 Sucres (g)",            min_value=0.0, value=10.0)
            fiber    = st.number_input("🌾 Fibres (g)",            min_value=0.0, value=2.0)
            proteins = st.number_input("💪 Protéines (g)",         min_value=0.0, value=5.0)
            salt     = st.number_input("🧂 Sel (g)",               min_value=0.0, value=0.5)
            additives= st.number_input("⚗️ Additifs",              min_value=0,   value=2, step=1)

        submitted = st.form_submit_button("🔮 Prédire le Nutri-Score", use_container_width=True)

    if submitted:
        vals = {
            "energy_100g": energy, "fat_100g": fat, "saturated-fat_100g": sat_fat,
            "carbohydrates_100g": carbs, "sugars_100g": sugars, "fiber_100g": fiber,
            "proteins_100g": proteins, "salt_100g": salt, "additives_n": additives,
        }
        grade = predict_nutriscore(model, vals)
        colors  = {"a":"#1fa37b","b":"#85bb2f","c":"#ffcc00","d":"#ff6600","e":"#ee3333"}
        labels  = {
            "a":"Excellent — Produit très sain ✅",
            "b":"Bon — Bonne qualité nutritionnelle 👍",
            "c":"Moyen — À consommer avec modération ⚠️",
            "d":"Médiocre — Qualité nutritionnelle faible 👎",
            "e":"Mauvais — À éviter ❌",
        }
        st.markdown(f"""
        <div style="background:{colors[grade]}22; border:2px solid {colors[grade]};
                    border-radius:12px; padding:1.5rem; text-align:center; margin-top:1rem">
            <div style="font-size:3rem; font-weight:700; color:{colors[grade]}">{grade.upper()}</div>
            <div style="font-size:1.1rem; color:#333; margin-top:0.5rem">{labels[grade]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(["a","b","c","d","e"].index(grade) / 4)
