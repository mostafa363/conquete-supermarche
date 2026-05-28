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
from src.fetcher import OpenFoodFactsFetcher
from src.repository import ProductRepository

NLP_MODEL_PATH  = MODEL_DIR / "nlp_nutriscore_clf.joblib"
NLP_METRICS_PATH = MODEL_DIR / "nlp_metrics.json"

_repo    = ProductRepository()
_fetcher = OpenFoodFactsFetcher()

ALLERGENS = {
    "Gluten":    ["gluten", "wheat", "farine de blé", "blé", "orge", "seigle"],
    "Lactose":   ["lait", "milk", "lactose", "beurre", "crème", "fromage"],
    "Fruits à coque": ["noix", "amande", "noisette", "cajou", "pistache", "noix de coco"],
    "Arachides": ["arachide", "cacahuète", "peanut"],
    "Soja":      ["soja", "soy"],
    "Oeufs":     ["oeuf", "egg"],
    "Poisson":   ["poisson", "fish", "saumon", "thon", "cabillaud"],
    "Céleri":    ["céleri", "celery"],
}

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
    .prod-img-box {
        height: 180px;
        overflow: hidden;
        border-radius: 8px;
        background: #f4f4f4;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .prod-img-box img {
        width: 100%;
        height: 180px;
        object-fit: cover;
    }
    .prod-card {
        min-height: 340px;
    }
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


def grade_badge(g: str) -> str:
    color = COLORS.get(g, "#888")
    return (
        f"<span style='background:{color};color:white;font-weight:700;"
        f"padding:2px 10px;border-radius:12px;font-size:0.95rem'>"
        f"Nutri-Score {g.upper()}</span>"
    )


def product_card(row, col):
    g = str(row.get("nutriscore_grade", "?")).lower()
    img = row.get("image_url", "")
    img_valid = img and str(img) not in ("nan", "None", "")

    img_html = (
        f"<div class='prod-img-box'><img src='{img}' "
        f"onerror=\"this.parentElement.innerHTML='<span style=font-size:3rem>🛒</span>'\"/></div>"
        if img_valid else
        "<div class='prod-img-box'><span style='font-size:3rem'>🛒</span></div>"
    )

    sugars_txt = f"🍬 {row['sugars_100g']:.1f}g" if row.get("sugars_100g") else ""
    salt_txt   = f"🧂 {row.get('salt_100g',0):.2f}g" if row.get("salt_100g") else ""
    additif_txt = f"⚗️ {int(row['additives_n'])} additif(s)" if row.get("additives_n") and int(row["additives_n"]) > 0 else ""
    color = COLORS.get(g, "#888")

    card_html = f"""
    <div style='border:1px solid #e0e0e0;border-radius:12px;padding:12px;
                background:white;min-height:340px;display:flex;flex-direction:column;gap:6px;'>
        {img_html}
        <div style='font-weight:700;font-size:0.95rem;line-height:1.3;min-height:2.6em;overflow:hidden'>
            {str(row['product_name'])[:50]}
        </div>
        <div style='color:#666;font-size:0.82rem'>🏷️ {str(row.get('brands','Inconnu'))[:30]}</div>
        <div>
            <span style='background:{color};color:white;font-weight:700;
                         padding:2px 10px;border-radius:12px;font-size:0.88rem'>
                Nutri-Score {g.upper()}
            </span>
        </div>
        <div style='color:#666;font-size:0.8rem;margin-top:auto'>
            {sugars_txt} {'|' if sugars_txt and salt_txt else ''} {salt_txt}
        </div>
        <div style='color:#888;font-size:0.8rem'>{additif_txt}</div>
    </div>
    """
    with col:
        st.markdown(card_html, unsafe_allow_html=True)


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
        "🔄 Substitution produits",
        "⚖️ Comparaison produits",
        "🗄️ Explorateur DuckDB",
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

    st.divider()
    st.markdown("**Allergènes à exclure**")
    allergens_selected = st.multiselect(
        "Exclure", list(ALLERGENS.keys()), default=[]
    )

# Appliquer filtres + allergènes
if not df.empty:
    dff = df[df["nutriscore_grade"].isin(ns_filter)].copy()
    if cat_filter != "Toutes":
        dff = dff[dff["categories_en"].str.contains(cat_filter, na=False)]
    for allergen in allergens_selected:
        keywords = ALLERGENS[allergen]
        mask = dff["ingredients_text"].str.lower().fillna("")
        for kw in keywords:
            mask = mask.str.contains(kw, na=False)
            dff = dff[~dff["ingredients_text"].str.lower().fillna("").str.contains(kw, na=False)]
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

    tab_search, tab_barcode = st.tabs(["🔎 Recherche textuelle", "📷 Recherche par code-barres"])

    with tab_search:
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

        if allergens_selected:
            st.info(f"Filtrage actif : **{', '.join(allergens_selected)}** exclus des ingrédients")

        st.info(f"**{len(results):,}** produit(s) trouvé(s)")

        if not results.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.head(30).iterrows()):
                product_card(row, cols[i % 3])

    with tab_barcode:
        st.subheader("📷 Recherche par code-barres EAN")
        st.info("Entrez un code EAN-13 pour retrouver un produit directement depuis Open Food Facts.")
        barcode = st.text_input("Code-barres EAN", placeholder="Ex: 3017620422003 (Nutella 400g)")

        if barcode.strip():
            # 1) ProductRepository — recherche en base DuckDB
            local_match = _repo.find_by_barcode(barcode.strip())

            if not local_match.empty:
                row = local_match.iloc[0]
                st.success("Produit trouvé dans la base locale (DuckDB)")
                c1, c2 = st.columns([1, 2])
                img = row.get("image_url", "")
                with c1:
                    if img and str(img) != "nan":
                        st.image(str(img), use_container_width=True)
                with c2:
                    st.markdown(f"### {row['product_name']}")
                    st.markdown(grade_badge(str(row.get("nutriscore_grade","?")).lower()), unsafe_allow_html=True)
                    st.markdown(f"**Marque :** {row.get('brands','N/A')}")
                    st.markdown(f"**Catégorie :** {row.get('categories_en','N/A')}")
                    if row.get("ingredients_text"):
                        st.markdown(f"**Ingrédients :** {str(row['ingredients_text'])[:300]}...")
            else:
                # 2) OpenFoodFactsFetcher — appel API si absent en base
                with st.spinner("Recherche sur Open Food Facts..."):
                    p = _fetcher.fetch_by_barcode(barcode.strip())
                    if p:
                        g = p.get("nutriscore_grade", "?").lower()
                        st.success("Produit trouvé sur Open Food Facts !")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            img = p.get("image_url", "")
                            if img:
                                st.image(img, use_container_width=True)
                        with c2:
                            st.markdown(f"### {p.get('product_name', 'Produit inconnu')}")
                            st.markdown(grade_badge(g), unsafe_allow_html=True)
                            st.markdown(f"**Marque :** {p.get('brands','N/A')}")
                            st.markdown(f"**Catégorie :** {p.get('categories','N/A')}")
                            nutriments = p.get("nutriments", {})
                            col1, col2 = st.columns(2)
                            col1.metric("⚡ Énergie", f"{nutriments.get('energy-kcal_100g',0):.0f} kcal")
                            col1.metric("🍬 Sucres", f"{nutriments.get('sugars_100g',0):.1f}g")
                            col2.metric("🧂 Sel", f"{nutriments.get('salt_100g',0):.2f}g")
                            col2.metric("💪 Protéines", f"{nutriments.get('proteins_100g',0):.1f}g")
                            ingr = p.get("ingredients_text", "")
                            if ingr:
                                st.markdown(f"**Ingrédients :** {ingr[:300]}...")
                    else:
                        st.warning("Produit non trouvé sur Open Food Facts.")

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

# ── PAGE 6 : Substitution ────────────────────────────────────────────────
elif page == "🔄 Substitution produits":
    st.header("🔄 Moteur de substitution")
    st.info(
        "Choisissez un produit de mauvaise qualité nutritionnelle (D ou E) "
        "et découvrez des alternatives plus saines dans la même catégorie."
    )

    if df.empty:
        st.warning("⚠️ Aucune donnée. Lance d'abord `python main.py`")
        st.stop()

    search_term = st.text_input("🔎 Rechercher un produit à remplacer", placeholder="Ex: chips, soda, biscuit...")

    candidates = df.copy()
    if search_term:
        candidates = candidates[
            candidates["product_name"].str.lower().str.contains(search_term.lower(), na=False)
        ]

    candidates = candidates[candidates["nutriscore_grade"].isin(["c","d","e"])]

    if candidates.empty:
        st.warning("Aucun produit trouvé avec un Nutri-Score C/D/E pour ce terme.")
    else:
        product_names = candidates["product_name"].tolist()
        selected_name = st.selectbox("Sélectionner le produit", product_names[:50])
        row = candidates[candidates["product_name"] == selected_name].iloc[0]

        g = str(row.get("nutriscore_grade", "?")).lower()
        st.markdown("---")
        st.markdown(f"**Produit sélectionné :** {row['product_name']}")
        st.markdown(grade_badge(g), unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Énergie", f"{row.get('energy_100g',0):.0f} kcal")
        col2.metric("🍬 Sucres", f"{row.get('sugars_100g',0):.1f}g")
        col3.metric("🧂 Sel", f"{row.get('salt_100g',0):.2f}g")

        st.markdown("---")
        st.subheader("✅ Alternatives plus saines")

        subs = _repo.find_substitutes(row, n=3)

        if subs.empty:
            st.warning("Aucune alternative trouvée dans la même catégorie. Essayez un autre produit.")
        else:
            sub_cols = st.columns(len(subs))
            for i, (_, sub_row) in enumerate(subs.iterrows()):
                sg = str(sub_row.get("nutriscore_grade","?")).lower()
                img = sub_row.get("image_url","")
                with sub_cols[i]:
                    with st.container(border=True):
                        if img and str(img) != "nan":
                            st.image(str(img), use_container_width=True)
                        else:
                            st.markdown(
                                "<div style='height:100px;background:#e8f5e9;border-radius:8px;"
                                "display:flex;align-items:center;justify-content:center;"
                                "font-size:2rem'>✅</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**{str(sub_row['product_name'])[:40]}**")
                        st.caption(f"🏷️ {sub_row.get('brands','')}")
                        st.markdown(grade_badge(sg), unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        col1.metric("⚡", f"{sub_row.get('energy_100g',0):.0f}")
                        col2.metric("🍬", f"{sub_row.get('sugars_100g',0):.1f}g")

# ── PAGE 7 : Comparaison ─────────────────────────────────────────────────
elif page == "⚖️ Comparaison produits":
    st.header("⚖️ Comparaison nutritionnelle de 2 produits")
    st.info("Sélectionnez deux produits pour comparer leurs valeurs nutritionnelles côte à côte.")

    if df.empty:
        st.warning("⚠️ Aucune donnée. Lance d'abord `python main.py`")
        st.stop()

    all_names = sorted(df["product_name"].dropna().unique().tolist())

    c1, c2 = st.columns(2)
    p1_name = c1.selectbox("🅰️ Produit 1", all_names, index=0)
    p2_name = c2.selectbox("🅱️ Produit 2", all_names, index=min(1, len(all_names)-1))

    row1 = df[df["product_name"] == p1_name].iloc[0]
    row2 = df[df["product_name"] == p2_name].iloc[0]

    g1 = str(row1.get("nutriscore_grade","?")).lower()
    g2 = str(row2.get("nutriscore_grade","?")).lower()

    # Header cards
    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True):
            img1 = row1.get("image_url","")
            if img1 and str(img1) != "nan":
                st.image(str(img1), width=200)
            st.markdown(f"### {row1['product_name'][:50]}")
            st.caption(f"🏷️ {row1.get('brands','')}")
            st.markdown(grade_badge(g1), unsafe_allow_html=True)
    with cc2:
        with st.container(border=True):
            img2 = row2.get("image_url","")
            if img2 and str(img2) != "nan":
                st.image(str(img2), width=200)
            st.markdown(f"### {row2['product_name'][:50]}")
            st.caption(f"🏷️ {row2.get('brands','')}")
            st.markdown(grade_badge(g2), unsafe_allow_html=True)

    st.divider()

    # Nutritional comparison table
    nutrients = {
        "⚡ Énergie (kcal)":       "energy_100g",
        "🥩 Graisses (g)":         "fat_100g",
        "🧈 Graisses saturées (g)": "saturated-fat_100g",
        "🍞 Glucides (g)":         "carbohydrates_100g",
        "🍬 Sucres (g)":           "sugars_100g",
        "🌾 Fibres (g)":           "fiber_100g",
        "💪 Protéines (g)":        "proteins_100g",
        "🧂 Sel (g)":              "salt_100g",
        "⚗️ Additifs":             "additives_n",
    }

    st.subheader("📋 Tableau comparatif")
    table_data = []
    for label, col in nutrients.items():
        v1 = float(row1.get(col, 0) or 0)
        v2 = float(row2.get(col, 0) or 0)
        winner = "🅰️" if v1 < v2 else ("🅱️" if v2 < v1 else "=")
        table_data.append({"Nutriment": label, p1_name[:25]: f"{v1:.2f}", p2_name[:25]: f"{v2:.2f}", "Meilleur": winner})
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Radar chart
    import plotly.graph_objects as go
    radar_cols = ["energy_100g","fat_100g","sugars_100g","fiber_100g","proteins_100g","salt_100g"]
    radar_labels = ["Énergie","Graisses","Sucres","Fibres","Protéines","Sel"]

    vals1 = [float(row1.get(c, 0) or 0) for c in radar_cols]
    vals2 = [float(row2.get(c, 0) or 0) for c in radar_cols]
    max_vals = [max(v1, v2, 1) for v1, v2 in zip(vals1, vals2)]
    norm1 = [v/m for v,m in zip(vals1, max_vals)]
    norm2 = [v/m for v,m in zip(vals2, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm1 + [norm1[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", name=row1["product_name"][:30],
        line_color=COLORS.get(g1, "#888"), opacity=0.7,
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm2 + [norm2[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", name=row2["product_name"][:30],
        line_color=COLORS.get(g2, "#444"), opacity=0.7,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="Comparaison nutritionnelle (valeurs normalisées)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 8 : DuckDB Explorer ─────────────────────────────────────────────
elif page == "🗄️ Explorateur DuckDB":
    st.header("🗄️ Explorateur DuckDB")
    st.info(
        "Interrogez directement la base DuckDB avec du SQL.  \n"
        "Table disponible : **`products`** avec toutes les colonnes nutritionnelles."
    )

    if _repo.count() == 0:
        st.warning("Base DuckDB non initialisée. Lance d'abord `python main.py`.")
        st.stop()

    with st.expander("📖 Schéma de la table `products`"):
        st.dataframe(_repo.schema(), use_container_width=True)

    examples = {
        "Top 10 produits les plus sucrés": "SELECT product_name, brands, nutriscore_grade, sugars_100g FROM products ORDER BY sugars_100g DESC LIMIT 10",
        "Distribution Nutri-Score":        "SELECT nutriscore_grade, COUNT(*) as n FROM products GROUP BY nutriscore_grade ORDER BY nutriscore_grade",
        "Produits sans additifs (Nutri-Score A)": "SELECT product_name, brands, energy_100g FROM products WHERE additives_n = 0 AND nutriscore_grade = 'a' LIMIT 20",
        "Marques avec le + de produits E": "SELECT brands, COUNT(*) as n_bad FROM products WHERE nutriscore_grade = 'e' GROUP BY brands ORDER BY n_bad DESC LIMIT 10",
        "Aliments riches en fibres":       "SELECT product_name, fiber_100g, nutriscore_grade FROM products WHERE fiber_100g > 5 ORDER BY fiber_100g DESC LIMIT 15",
    }

    col_ex, col_run = st.columns([3,1])
    selected_ex = col_ex.selectbox("Exemples de requêtes", ["— Requête personnalisée —"] + list(examples.keys()))

    default_sql = examples.get(selected_ex, "SELECT * FROM products LIMIT 10")
    sql_query = st.text_area("Requête SQL", value=default_sql, height=120)

    if st.button("▶ Exécuter", type="primary"):
        with st.spinner("Exécution..."):
            result = _repo.execute_sql(sql_query)
        if "error" in result.columns:
            st.error(result["error"].iloc[0])
        else:
            st.success(f"{len(result):,} ligne(s) retournée(s)")
            st.dataframe(result, use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV", csv, "query_result.csv", "text/csv")
