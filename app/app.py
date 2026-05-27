import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_CSV_PATH, FEATURE_COLUMNS, NUTRISCORE_LABELS, MODEL_DIR
from src.analysis import (
    plot_nutriscore_distribution, plot_top_brands_controverses,
    plot_correlation_matrix, plot_nutriscore_by_category,
    plot_additives_vs_nutriscore, plot_sugar_salt_scatter,
    describe_dataset,
)
from src.model import load_model, predict_nutriscore

NLP_MODEL_PATH  = MODEL_DIR / "nlp_nutriscore_clf.joblib"
NLP_METRICS_PATH = MODEL_DIR / "nlp_metrics.json"

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
</style>
""", unsafe_allow_html=True)

COLORS  = {"a":"#1fa37b","b":"#85bb2f","c":"#ffcc00","d":"#ff6600","e":"#ee3333"}
LABELS  = {
    "a":"Excellent — Produit très sain ✅",
    "b":"Bon — Bonne qualité nutritionnelle 👍",
    "c":"Moyen — À consommer avec modération ⚠️",
    "d":"Médiocre — Qualité nutritionnelle faible 👎",
    "e":"Mauvais — À éviter ❌",
}


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


@st.cache_resource
def get_nlp_model():
    try:
        from src.nlp_model import load_nlp_model
        return load_nlp_model()
    except Exception:
        return None, None


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
        "🧠 Prédiction NLP (Hugging Face)",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Filtres globaux**")

    ns_filter = st.multiselect(
        "Nutri-Score", ["a","b","c","d","e"],
        default=["a","b","c","d","e"], format_func=str.upper,
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
    c1.metric("🛒 Produits",      f"{s['n_products']:,}")
    c2.metric("🏷️ Marques",       f"{s['n_brands']:,}")
    c3.metric("📂 Catégories",    f"{s['n_categories']:,}")
    c4.metric("⚗️ Additifs moy.", f"{s['avg_additives']:.1f}")

    st.divider()
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_nutriscore_distribution(dff), use_container_width=True)
    c2.plotly_chart(plot_additives_vs_nutriscore(dff), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_top_brands_controverses(dff), use_container_width=True)
    c2.plotly_chart(plot_nutriscore_by_category(dff),  use_container_width=True)

# ── PAGE 2 : Recherche ───────────────────────────────────────────────────
elif page == "🔍 Recherche produits":
    st.header("🔍 Recherche de produits")

    c1, c2, c3 = st.columns([3, 2, 1])
    term  = c1.text_input("Nom du produit", placeholder="Ex: Nutella, yaourt, coca...")
    grade = c2.selectbox("Nutri-Score", ["Tous","A","B","C","D","E"])
    cat   = c3.text_input("Catégorie", placeholder="dairy...")

    results = dff.copy()
    if term:
        results = results[results["product_name"].str.lower().str.contains(term.lower(), na=False)]
    if grade != "Tous":
        results = results[results["nutriscore_grade"] == grade.lower()]
    if cat:
        results = results[results["categories_en"].str.lower().str.contains(cat.lower(), na=False)]

    st.info(f"**{len(results):,}** produit(s) trouvé(s)")

    if not results.empty:
        grade_emoji = {"a":"🟢","b":"🟡","c":"🟠","d":"🔴","e":"🔴"}
        grade_color = {"a":"#1fa37b","b":"#85bb2f","c":"#ffcc00","d":"#ff6600","e":"#ee3333"}
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.head(30).iterrows()):
            g = str(row.get("nutriscore_grade","?")).lower()
            img = row.get("image_url","")
            with cols[i % 3]:
                with st.container(border=True):
                    if img and str(img) != "nan":
                        st.image(str(img), use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:140px;background:#f0f0f0;border-radius:8px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "font-size:2.5rem'>🛒</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**{str(row['product_name'])[:45]}**")
                    st.caption(f"🏷️ {row.get('brands','Inconnu')}")
                    color = grade_color.get(g, "#888")
                    st.markdown(
                        f"<span style='background:{color};color:white;font-weight:700;"
                        f"padding:2px 10px;border-radius:12px;font-size:0.95rem'>"
                        f"Nutri-Score {g.upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    if row.get("sugars_100g"):
                        st.caption(f"🍬 Sucres : {row['sugars_100g']:.1f}g | 🧂 Sel : {row.get('salt_100g',0):.2f}g")
                    if row.get("additives_n") and int(row["additives_n"]) > 0:
                        st.caption(f"⚗️ {int(row['additives_n'])} additif(s)")

# ── PAGE 3 : Analyses ────────────────────────────────────────────────────
elif page == "📊 Analyses détaillées":
    st.header("📊 Analyses nutritionnelles")

    t1, t2, t3, t4 = st.tabs(["📊 Distributions", "🔗 Corrélations", "🍬 Sucres & Sel", "🏆 Feature Importance"])

    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_nutriscore_distribution(dff), use_container_width=True)
        c2.plotly_chart(plot_top_brands_controverses(dff), use_container_width=True)
        st.plotly_chart(plot_nutriscore_by_category(dff),  use_container_width=True)
    with t2:
        st.plotly_chart(plot_correlation_matrix(dff),      use_container_width=True)
    with t3:
        st.plotly_chart(plot_sugar_salt_scatter(dff),      use_container_width=True)
    with t4:
        st.subheader("Importance des variables — XGBoost")
        if model is not None:
            import plotly.express as px
            xgb_model = model.named_steps["model"]
            fi = xgb_model.feature_importances_
            cols_fi = [c for c in FEATURE_COLUMNS if c in dff.columns]
            fig = px.bar(
                x=fi[:len(cols_fi)], y=cols_fi,
                orientation="h",
                labels={"x": "Importance", "y": "Variable"},
                title="Feature Importance XGBoost",
                color=fi[:len(cols_fi)],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

            import json
            metrics_path = MODEL_DIR / "metrics.json"
            if metrics_path.exists():
                m = json.load(open(metrics_path))
                st.info(f"**XGBoost** — Accuracy : {m['accuracy']*100:.1f}% | F1 : {m['f1_score']:.4f}")

            nlp_path = MODEL_DIR / "nlp_metrics.json"
            if nlp_path.exists():
                m2 = json.load(open(nlp_path))
                st.info(f"**NLP (Hugging Face)** — Accuracy : {m2['accuracy']*100:.1f}% | F1 : {m2['f1_score']:.4f} | Modèle : `{m2['hf_model']}`")
        else:
            st.warning("Modèle non chargé. Lance d'abord `python main.py`")

# ── PAGE 4 : Prédiction XGBoost ──────────────────────────────────────────
elif page == "🤖 Prédiction IA":
    st.header("🤖 Prédiction Nutri-Score — XGBoost (valeurs nutritionnelles)")
    st.info("Renseignez les valeurs nutritionnelles pour 100g.")

    if model is None:
        st.error("Modèle non disponible. Lance `python main.py`")
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
        st.markdown(f"""
        <div style="background:{COLORS[grade]}22; border:2px solid {COLORS[grade]};
                    border-radius:12px; padding:1.5rem; text-align:center; margin-top:1rem">
            <div style="font-size:3rem; font-weight:700; color:{COLORS[grade]}">{grade.upper()}</div>
            <div style="font-size:1.1rem; color:#333; margin-top:0.5rem">{LABELS[grade]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(["a","b","c","d","e"].index(grade) / 4)

# ── PAGE 5 : Prédiction NLP ──────────────────────────────────────────────
elif page == "🧠 Prédiction NLP (Hugging Face)":
    st.header("🧠 Prédiction Nutri-Score — NLP Hugging Face")
    st.info(
        "Modèle : **paraphrase-multilingual-MiniLM-L12-v2** (Hugging Face sentence-transformers)  \n"
        "Entrez le nom et/ou la liste des ingrédients du produit — le modèle NLP prédit le Nutri-Score."
    )

    with st.expander("ℹ️ Comment ça fonctionne ?"):
        st.markdown("""
        1. Le texte (nom + ingrédients) est converti en vecteur de 384 dimensions via un modèle Transformer multilingue
        2. Ce vecteur est classifié par une Régression Logistique entraînée sur les données Open Food Facts
        3. Le résultat est le Nutri-Score prédit A → E avec les probabilités par classe
        """)

    embedder, nlp_clf = get_nlp_model()

    if nlp_clf is None:
        st.warning("Modèle NLP non disponible. Lance `python main.py` pour l'entraîner (~5 min).")
        st.stop()

    product_name = st.text_input("🏷️ Nom du produit", placeholder="Ex: Biscuits chocolat noisette...")
    ingredients  = st.text_area("📋 Liste des ingrédients",
                                placeholder="Ex: farine de blé, sucre, huile de palme, cacao 5%...",
                                height=120)

    if st.button("🔮 Prédire via NLP", use_container_width=True):
        text = (product_name + " " + ingredients).strip()
        if len(text) < 5:
            st.error("Entrez au moins le nom du produit ou les ingrédients.")
        else:
            from src.nlp_model import predict_nutriscore_nlp
            grade, probas = predict_nutriscore_nlp(embedder, nlp_clf, text)

            st.markdown(f"""
            <div style="background:{COLORS[grade]}22; border:2px solid {COLORS[grade]};
                        border-radius:12px; padding:1.5rem; text-align:center; margin-top:1rem">
                <div style="font-size:3rem; font-weight:700; color:{COLORS[grade]}">{grade.upper()}</div>
                <div style="font-size:1.1rem; color:#333; margin-top:0.5rem">{LABELS[grade]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probabilités par classe")
            import plotly.graph_objects as go
            grades_order = ["a","b","c","d","e"]
            fig = go.Figure(go.Bar(
                x=[g.upper() for g in grades_order],
                y=[probas.get(g, 0) for g in grades_order],
                marker_color=[COLORS[g] for g in grades_order],
                text=[f"{probas.get(g,0)*100:.1f}%" for g in grades_order],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(title="Probabilité", range=[0, 1]),
                xaxis_title="Nutri-Score", showlegend=False,
                plot_bgcolor="white", height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
