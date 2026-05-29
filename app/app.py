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
from src.auth import (
    init_auth_tables, register_user, login_user,
    save_substitute, get_user_substitutes, delete_substitute,
)

NLP_MODEL_PATH   = MODEL_DIR / "nlp_nutriscore_clf.joblib"
NLP_METRICS_PATH = MODEL_DIR / "nlp_metrics.json"

# Initialiser les tables auth au démarrage
init_auth_tables()

_repo    = ProductRepository()
_fetcher = OpenFoodFactsFetcher()

# ── Session state auth ──────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ───────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(145deg, #f7fffe 0%, #eef9f4 40%, #f0f5ff 100%);
    background-attachment: fixed;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(29,122,94,0.35); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1D7A5E; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1c18 0%, #10261f 55%, #0a1e17 100%) !important;
    border-right: 1px solid rgba(29,122,94,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.35) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem 2rem !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #a8d5c2 !important; }
[data-testid="stSidebar"] h2 {
    font-size: 1.5rem !important; font-weight: 900 !important;
    color: #ffffff !important; letter-spacing: -0.5px;
}

/* ── Nav items ───────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio > div > div { gap: 3px !important; display:flex !important; flex-direction:column !important; }

/* Masquer le cercle radio natif sans casser les clics */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    appearance: none !important; -webkit-appearance: none !important;
    width: 8px !important; height: 8px !important;
    border-radius: 50% !important; flex-shrink: 0 !important;
    background: rgba(29,122,94,0.4) !important;
    border: none !important; outline: none !important;
    margin-right: 10px !important; margin-left: 4px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked {
    background: #34d399 !important;
    box-shadow: 0 0 8px rgba(52,211,153,0.7) !important;
    width: 9px !important; height: 9px !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 12px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    border: 1px solid transparent !important;
    color: #8bbfae !important;
    line-height: 1.4 !important;
    margin: 1px 0 !important;
}
[data-testid="stSidebar"] .stRadio label p {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: inherit !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

/* Hover */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(29,122,94,0.18) !important;
    color: #e0f5ed !important;
    border-color: rgba(29,122,94,0.28) !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] .stRadio label:hover p { color: #e0f5ed !important; }
[data-testid="stSidebar"] .stRadio label:hover input[type="radio"] {
    background: #34d399 !important;
    box-shadow: 0 0 6px rgba(52,211,153,0.55) !important;
}

/* Actif / sélectionné */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(135deg, rgba(29,122,94,0.38) 0%, rgba(39,168,124,0.22) 100%) !important;
    color: #ffffff !important;
    border-color: rgba(52,211,153,0.35) !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(29,122,94,0.25), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] hr { border-color: rgba(29,122,94,0.2) !important; }
[data-testid="stSidebar"] .stSuccess {
    background: rgba(29,122,94,0.2) !important;
    border: 1px solid rgba(29,122,94,0.4) !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] .stInfo {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.07) !important;
    color: #a8d5c2 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(239,68,68,0.18) !important;
    color: #fca5a5 !important;
    border-color: rgba(239,68,68,0.3) !important;
    transform: none !important;
}

/* Caption sous le logo */
[data-testid="stSidebar"] .stCaption { color: #4d8c78 !important; font-size:.75rem !important; }

/* Label filtres */
[data-testid="stSidebar"] .stMarkdown strong { color: #c8e6dc !important; font-size:.72rem !important; letter-spacing:1.5px !important; text-transform:uppercase !important; }

/* ── Main block ──────────────────────────────────────────── */
.main .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1300px; }

/* ── Animations ──────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn    { from { opacity:0; } to { opacity:1; } }
@keyframes slideLeft {
    from { opacity:0; transform:translateX(-16px); }
    to   { opacity:1; transform:translateX(0); }
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(29,122,94,0.3); }
    50%      { box-shadow: 0 0 0 8px rgba(29,122,94,0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}

.main .block-container > div > div {
    animation: fadeInUp 0.45s ease both;
}

/* ── Headings ────────────────────────────────────────────── */
h1 {
    font-weight: 900 !important; letter-spacing: -1.2px !important;
    background: linear-gradient(135deg, #1D7A5E 0%, #34d399 60%, #059669 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2 { font-weight: 700 !important; color: #1a2e28 !important; letter-spacing: -.4px !important; }
h3 { font-weight: 600 !important; color: #2d4a44 !important; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 10px 22px !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    border: none !important;
    background: linear-gradient(135deg, #1D7A5E, #27a87c) !important;
    color: #fff !important;
    box-shadow: 0 4px 18px rgba(29,122,94,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(29,122,94,0.42) !important;
    filter: brightness(1.06) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ──────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    border-radius: 12px !important; border: 2px solid #e2ecf0 !important;
    padding: 10px 16px !important; font-size: 0.9rem !important;
    transition: all .2s !important; background: #fff !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #1D7A5E !important;
    box-shadow: 0 0 0 4px rgba(29,122,94,0.12) !important;
    outline: none !important;
}

/* ── Select ──────────────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {
    border-radius: 12px !important; border: 2px solid #e2ecf0 !important;
    transition: border-color .2s !important;
}
[data-baseweb="select"] > div:first-child:focus-within {
    border-color: #1D7A5E !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,.85); border-radius: 14px;
    padding: 5px; border: 1.5px solid #e8f5f0;
    backdrop-filter: blur(8px); gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.85rem !important; transition: all .2s !important;
    border: none !important; color: #64748b !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1D7A5E, #27a87c) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(29,122,94,0.3) !important;
}

/* ── Metrics ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #fff; border-radius: 18px;
    padding: 1.3rem 1.6rem;
    border: 1.5px solid #e8f5f0;
    box-shadow: 0 4px 22px rgba(0,0,0,0.055);
    transition: all .3s ease; animation: fadeInUp .5s ease both;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(29,122,94,0.13);
    border-color: #a7f3d0;
}
[data-testid="stMetric"] label {
    color: #94a3b8 !important; font-size: .76rem !important;
    font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: .8px !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 800 !important; color: #1a2e28 !important;
}

/* ── Alerts ──────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 14px !important; border: none !important; }
.stSuccess { background: linear-gradient(135deg,#d1fae5,#a7f3d0) !important; border-left: 4px solid #1D7A5E !important; }
.stInfo    { background: linear-gradient(135deg,#dbeafe,#bfdbfe) !important; border-left: 4px solid #3b82f6 !important; }
.stWarning { background: linear-gradient(135deg,#fef3c7,#fde68a) !important; border-left: 4px solid #f59e0b !important; }
.stError   { background: linear-gradient(135deg,#fee2e2,#fecaca) !important; border-left: 4px solid #ef4444 !important; }

/* ── DataFrame ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important; overflow: hidden !important;
    border: 1.5px solid #e8f5f0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
}

/* ── Containers ──────────────────────────────────────────── */
[data-testid="stContainer"] { border-radius: 18px !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #e8f5f0 !important; margin: 1.5rem 0 !important; }

/* ──────────────────────── CUSTOM CLASSES ──────────────────── */

/* Hero */
.hero-wrap {
    padding: 2.5rem 0 1.5rem;
    animation: fadeInUp .6s ease;
}
.hero-eyebrow {
    font-size: .72rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #1D7A5E;
    background: rgba(29,122,94,.1); display: inline-block;
    padding: 4px 14px; border-radius: 100px; margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.4rem; font-weight: 900; line-height: 1.05;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #1D7A5E 0%, #34d399 55%, #059669 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .6rem;
}
.hero-sub {
    font-size: 1.05rem; color: #64748b; font-weight: 400;
    max-width: 640px; line-height: 1.7;
}
.hero-stats {
    display: flex; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;
}
.hero-stat { display: flex; flex-direction: column; gap: 2px; }
.hero-stat-val { font-size: 1.6rem; font-weight: 800; color: #1a2e28; letter-spacing: -.5px; }
.hero-stat-lbl { font-size: .72rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .8px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 2rem; }
.kpi-card {
    background: #fff; border-radius: 20px; padding: 1.4rem 1.6rem;
    border: 1.5px solid #e8f5f0;
    box-shadow: 0 4px 24px rgba(0,0,0,.055);
    transition: all .3s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
    animation: fadeInUp .5s ease both;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg, #1D7A5E, #34d399);
    border-radius: 20px 20px 0 0;
}
.kpi-card:hover { transform: translateY(-5px); box-shadow: 0 24px 50px rgba(29,122,94,.14); border-color: #a7f3d0; }
.kpi-icon { font-size: 1.8rem; margin-bottom: .5rem; }
.kpi-value { font-size: 2rem; font-weight: 900; color: #1a2e28; letter-spacing: -.5px; line-height: 1; }
.kpi-label { font-size: .72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .8px; margin-top: .35rem; }

/* Page header */
.page-hdr {
    background: linear-gradient(135deg, rgba(29,122,94,.07) 0%, rgba(52,211,153,.04) 100%);
    border-radius: 20px; padding: 1.8rem 2.2rem; margin-bottom: 1.8rem;
    border: 1.5px solid rgba(29,122,94,.1);
    animation: fadeIn .45s ease;
}
.page-hdr h2 { margin: 0 0 .3rem; color: #1a2e28 !important; }
.page-hdr p  { margin: 0; color: #64748b; font-size: .9rem; }

/* Product card */
.pc {
    background: #fff; border-radius: 20px; padding: 14px;
    border: 1.5px solid #f0f5f2;
    box-shadow: 0 2px 18px rgba(0,0,0,.055);
    transition: all .3s cubic-bezier(.4,0,.2,1);
    display: flex; flex-direction: column; gap: 8px; height: 100%;
    animation: fadeInUp .4s ease both;
    position: relative; overflow: hidden;
}
.pc::after {
    content:''; position:absolute; inset:0;
    background: linear-gradient(135deg, transparent 65%, rgba(29,122,94,.04) 100%);
    opacity:0; transition: opacity .3s;
}
.pc:hover { transform: translateY(-7px) scale(1.015); box-shadow: 0 28px 55px rgba(0,0,0,.12); border-color: #a7f3d0; }
.pc:hover::after { opacity: 1; }
.pc-img {
    width:100%; height:160px; object-fit:cover;
    border-radius: 12px; background: linear-gradient(135deg,#f0fdf4,#dcfce7);
}
.pc-img-placeholder {
    width:100%; height:160px; border-radius:12px;
    background: linear-gradient(135deg,#f0fdf4,#e8f5e9);
    display:flex; align-items:center; justify-content:center; font-size:3rem;
}
.pc-name { font-weight:700; font-size:.92rem; line-height:1.35; color:#1a2e28; min-height:2.5em; overflow:hidden; }
.pc-brand { color:#94a3b8; font-size:.78rem; font-weight:500; }
.pc-nutrients { color:#64748b; font-size:.78rem; display:flex; gap:8px; flex-wrap:wrap; margin-top:auto; }

/* Nutri-Score badge */
.ns-badge {
    display:inline-flex; align-items:center; gap:5px;
    padding: 5px 13px; border-radius:100px;
    font-weight:800; font-size:.8rem; letter-spacing:.3px;
    box-shadow: 0 3px 10px rgba(0,0,0,.18);
    transition: all .2s; color: #fff;
}
.ns-badge:hover { transform: scale(1.06); box-shadow: 0 5px 18px rgba(0,0,0,.22); }

/* Section label */
.sect {
    font-size:.72rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    color:#1D7A5E; margin-bottom:.8rem;
    display:flex; align-items:center; gap:8px;
}
.sect::after { content:''; flex:1; height:1.5px; background:linear-gradient(90deg,#e8f5f0,transparent); }

/* ── Sidebar filter section headers ─────────────────────── */
.sb-section-hdr {
    display: flex; align-items: center; gap: 7px;
    font-size: .68rem !important; font-weight: 800 !important;
    letter-spacing: 1.6px !important; text-transform: uppercase !important;
    color: #3d8c72 !important;
    margin: 1rem 0 .55rem 2px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid rgba(29,122,94,0.2) !important;
}
.sb-section-icon { font-size: .8rem; }

/* ── Sidebar widgets (multiselect / selectbox) ───────────── */
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label {
    font-size: .75rem !important; font-weight: 600 !important;
    color: #5fa08a !important; letter-spacing: .4px !important;
    margin-bottom: 3px !important;
}

/* Select container */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(29,122,94,0.3) !important;
    border-radius: 10px !important;
    transition: border-color .2s, box-shadow .2s !important;
    min-height: 38px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color: rgba(52,211,153,0.55) !important;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.1) !important;
}

/* Placeholder & value text */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div {
    color: #a8d5c2 !important; font-size: .82rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #c8e6dc !important;
}

/* Dropdown arrow icon */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #4d8c78 !important;
}

/* Multiselect chips */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(29,122,94,0.32) !important;
    border: 1px solid rgba(52,211,153,0.28) !important;
    border-radius: 8px !important;
    padding: 2px 6px !important;
    margin: 2px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #a7f3d0 !important;
    font-size: .75rem !important;
    font-weight: 700 !important;
}
/* X close sur les chips */
[data-testid="stSidebar"] [data-baseweb="tag"] [role="button"],
[data-testid="stSidebar"] [data-baseweb="tag"] button {
    color: #6ee7b7 !important;
    opacity: .7 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] [role="button"]:hover {
    opacity: 1 !important; color: #f87171 !important;
}

/* ── Legacy compat ───────────────────────────────────────── */
.big-title  { font-size:2.4rem; font-weight:900; color:#1D7A5E; margin-bottom:0; }
.subtitle   { color:#888; margin-top:0; margin-bottom:2rem; }
.prod-img-box {
    height:165px; overflow:hidden; border-radius:12px;
    background: linear-gradient(135deg,#f0fdf4,#e8f5e9);
    display:flex; align-items:center; justify-content:center; margin-bottom:.5rem;
}
.prod-img-box img { width:100%; height:165px; object-fit:cover; }
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
    icons = {"a": "🌿", "b": "✅", "c": "⚠️", "d": "🔶", "e": "❌"}
    icon = icons.get(g, "")
    return (
        f"<span class='ns-badge' style='background:{color}'>"
        f"{icon} Nutri-Score {g.upper()}</span>"
    )


def product_card(row, col):
    g = str(row.get("nutriscore_grade", "?")).lower()
    img = row.get("image_url", "")
    img_valid = img and str(img) not in ("nan", "None", "")

    img_html = (
        f"<img class='pc-img' src='{img}' "
        f"onerror=\"this.outerHTML='<div class=\\'pc-img-placeholder\\'>🛒</div>'\"/>"
        if img_valid else
        "<div class='pc-img-placeholder'>🛒</div>"
    )

    sugars_txt  = f"🍬 {row['sugars_100g']:.1f}g" if row.get("sugars_100g") else ""
    salt_txt    = f"🧂 {row.get('salt_100g',0):.2f}g" if row.get("salt_100g") else ""
    additif_txt = f"⚗️ {int(row['additives_n'])}add" if row.get("additives_n") and int(row["additives_n"]) > 0 else ""
    color = COLORS.get(g, "#888")
    icons = {"a": "🌿", "b": "✅", "c": "⚠️", "d": "🔶", "e": "❌"}
    icon = icons.get(g, "")

    nutrients_parts = [p for p in [sugars_txt, salt_txt, additif_txt] if p]
    nutrients_html = "".join(f"<span>{p}</span>" for p in nutrients_parts)

    card_html = f"""
    <div class='pc'>
        {img_html}
        <div class='pc-name'>{str(row['product_name'])[:55]}</div>
        <div class='pc-brand'>🏷️ {str(row.get('brands','Inconnu'))[:30]}</div>
        <span class='ns-badge' style='background:{color};align-self:flex-start'>
            {icon} {g.upper()}
        </span>
        <div class='pc-nutrients'>{nutrients_html}</div>
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

    # Statut connexion
    if st.session_state.user:
        st.success(f"👤 {st.session_state.user['username']}")
        if st.button("Déconnexion", use_container_width=True):
            st.session_state.user = None
            st.rerun()
    else:
        st.info("👤 Non connecté")

    st.divider()

    pages_base = [
        "🏠 Dashboard",
        "🔍 Recherche produits",
        "📊 Analyses détaillées",
        "🤖 Prédiction IA",
        "🧠 Prédiction NLP (Hugging Face)",
        "🔄 Substitution produits",
        "⚖️ Comparaison produits",
        "🗄️ Explorateur DuckDB",
        "🏗️ Architecture Big Data",
        "👤 Connexion / Inscription",
    ]
    if st.session_state.user:
        pages_base.append("💾 Mes substituts sauvegardés")

    page = st.radio("", pages_base, label_visibility="collapsed")

    st.markdown("""
    <div class="sb-section-hdr">
        <span class="sb-section-icon">⚙️</span>
        <span>Filtres globaux</span>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("""
    <div class="sb-section-hdr" style="margin-top:.6rem">
        <span class="sb-section-icon">🚫</span>
        <span>Allergènes à exclure</span>
    </div>
    """, unsafe_allow_html=True)
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
    if df.empty:
        st.warning("⚠️ Aucune donnée. Lance d'abord `python main.py`")
        st.stop()

    s = describe_dataset(dff)

    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">OPEN FOOD FACTS · ANALYSE IA</div>
        <div class="hero-title">Mangez mieux,<br>chaque jour.</div>
        <p class="hero-sub">
            SoGood analyse {s['n_products']:,} produits alimentaires avec un moteur XGBoost
            et NLP multilingue pour vous aider à faire les meilleurs choix nutritionnels.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-val">{s['n_products']:,}</span>
                <span class="hero-stat-lbl">Produits</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-val">{s['n_brands']:,}</span>
                <span class="hero-stat-lbl">Marques</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-val">{s['n_categories']:,}</span>
                <span class="hero-stat-lbl">Catégories</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-val">93.6%</span>
                <span class="hero-stat-lbl">Précision IA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pct_ab = 100 * len(dff[dff["nutriscore_grade"].isin(["a","b"])]) / max(len(dff), 1)
    avg_sugar = dff["sugars_100g"].mean() if "sugars_100g" in dff.columns else 0
    avg_salt  = dff["salt_100g"].mean()   if "salt_100g"   in dff.columns else 0

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">🛒</div>
            <div class="kpi-value">{s['n_products']:,}</div>
            <div class="kpi-label">Produits analysés</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🌿</div>
            <div class="kpi-value">{pct_ab:.0f}%</div>
            <div class="kpi-label">Nutri-Score A ou B</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🍬</div>
            <div class="kpi-value">{avg_sugar:.1f}g</div>
            <div class="kpi-label">Sucres moy. / 100g</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⚗️</div>
            <div class="kpi-value">{s['avg_additives']:.1f}</div>
            <div class="kpi-label">Additifs moy.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_nutriscore_distribution(dff), use_container_width=True)
    c2.plotly_chart(plot_additives_vs_nutriscore(dff), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_top_brands_controverses(dff), use_container_width=True)
    c2.plotly_chart(plot_nutriscore_by_category(dff),  use_container_width=True)

# ── PAGE 2 : Recherche ───────────────────────────────────────────────────
elif page == "🔍 Recherche produits":
    st.markdown("""
    <div class="page-hdr">
        <h2>🔍 Recherche de produits</h2>
        <p>Trouvez un produit par nom, catégorie ou code-barres EAN parmi les 52 000+ références.</p>
    </div>
    """, unsafe_allow_html=True)

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
                try:
                    with st.spinner("Recherche sur Open Food Facts..."):
                        p = _fetcher.fetch_by_barcode(barcode.strip())
                    if p:
                        g = p.get("nutriscore_grade", "?").lower()
                        if g not in COLORS:
                            g = "?"
                        st.success("Produit trouvé sur Open Food Facts !")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            img = p.get("image_url", "")
                            if img:
                                st.image(img, use_container_width=True)
                        with c2:
                            st.markdown(f"### {p.get('product_name', 'Produit inconnu')}")
                            if g in COLORS:
                                st.markdown(grade_badge(g), unsafe_allow_html=True)
                            st.markdown(f"**Marque :** {p.get('brands','N/A')}")
                            st.markdown(f"**Catégorie :** {p.get('categories','N/A')}")
                            nutriments = p.get("nutriments", {})
                            col1, col2 = st.columns(2)
                            energy = nutriments.get("energy-kcal_100g") or nutriments.get("energy_100g", 0)
                            col1.metric("⚡ Énergie",   f"{float(energy or 0):.0f} kcal")
                            col1.metric("🍬 Sucres",    f"{float(nutriments.get('sugars_100g', 0) or 0):.1f}g")
                            col2.metric("🧂 Sel",       f"{float(nutriments.get('salt_100g', 0) or 0):.2f}g")
                            col2.metric("💪 Protéines", f"{float(nutriments.get('proteins_100g', 0) or 0):.1f}g")
                            ingr = p.get("ingredients_text", "")
                            if ingr:
                                st.markdown(f"**Ingrédients :** {ingr[:300]}...")
                    else:
                        st.warning(
                            f"Produit `{barcode.strip()}` introuvable sur Open Food Facts.  \n"
                            "Vérifie que le code EAN est correct (13 chiffres sans espace)."
                        )
                except Exception as e:
                    st.error(f"Erreur de connexion à l'API Open Food Facts : {e}")

# ── PAGE 3 : Analyses ────────────────────────────────────────────────────
elif page == "📊 Analyses détaillées":
    st.markdown("""
    <div class="page-hdr">
        <h2>📊 Analyses nutritionnelles</h2>
        <p>Distributions, corrélations et importance des variables du modèle XGBoost.</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📊 Distributions", "🔗 Corrélations", "🍬 Sucres & Sel", "🏆 Feature Importance", "🎯 Évaluation modèles"])

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

    with t5:
        st.subheader("🎯 Matrice de confusion — XGBoost")
        if model is not None and not df.empty:
            import plotly.figure_factory as ff
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix, classification_report

            feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            df_eval = df.dropna(subset=feat_cols + ["nutriscore_grade"])
            X = df_eval[feat_cols]
            y = df_eval["nutriscore_grade"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)

            labels_order = ["a","b","c","d","e"]
            cm = confusion_matrix(y_test, y_pred, labels=labels_order)
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            fig_cm = ff.create_annotated_heatmap(
                z=cm_pct.tolist(),
                x=[g.upper() for g in labels_order],
                y=[g.upper() for g in labels_order],
                annotation_text=[[f"{v:.0f}%" for v in row] for row in cm_pct],
                colorscale="Greens", showscale=True,
            )
            fig_cm.update_layout(
                title="Matrice de confusion XGBoost (% par classe réelle)",
                xaxis_title="Prédit", yaxis_title="Réel",
                xaxis=dict(side="bottom"), height=420,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            report = classification_report(y_test, y_pred, target_names=[g.upper() for g in labels_order], output_dict=True)
            report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
            st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None), use_container_width=True)

            st.markdown("---")
            st.subheader("🧠 Matrice de confusion — NLP (Hugging Face)")
            embedder, nlp_clf = get_nlp_model()
            if nlp_clf is not None:
                df_nlp = df.dropna(subset=["product_name","nutriscore_grade"])
                df_nlp = df_nlp.sample(min(3000, len(df_nlp)), random_state=42)
                texts = (df_nlp["product_name"].fillna("") + " " + df_nlp.get("ingredients_text", pd.Series([""] * len(df_nlp))).fillna("")).tolist()
                _, X_t, _, y_t = train_test_split(texts, df_nlp["nutriscore_grade"].tolist(), test_size=0.2, random_state=42)
                with st.spinner("Encodage NLP en cours..."):
                    X_emb = embedder.encode(X_t, show_progress_bar=False)
                y_p = nlp_clf.predict(X_emb)
                cm2 = confusion_matrix(y_t, y_p, labels=labels_order)
                cm2_pct = cm2.astype(float) / (cm2.sum(axis=1, keepdims=True) + 1e-9) * 100
                fig_cm2 = ff.create_annotated_heatmap(
                    z=cm2_pct.tolist(),
                    x=[g.upper() for g in labels_order],
                    y=[g.upper() for g in labels_order],
                    annotation_text=[[f"{v:.0f}%" for v in row] for row in cm2_pct],
                    colorscale="Blues", showscale=True,
                )
                fig_cm2.update_layout(
                    title="Matrice de confusion NLP (% par classe réelle)",
                    xaxis_title="Prédit", yaxis_title="Réel",
                    xaxis=dict(side="bottom"), height=420,
                )
                st.plotly_chart(fig_cm2, use_container_width=True)
            else:
                st.warning("Modèle NLP non disponible.")
        else:
            st.warning("Données ou modèle non disponibles.")

# ── PAGE 4 : Prédiction XGBoost ──────────────────────────────────────────
elif page == "🤖 Prédiction IA":
    st.markdown("""
    <div class="page-hdr">
        <h2>🤖 Prédiction Nutri-Score — XGBoost</h2>
        <p>Renseignez les valeurs nutritionnelles pour 100g et obtenez une prédiction instantanée.</p>
    </div>
    """, unsafe_allow_html=True)

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
        icons = {"a": "🌿", "b": "✅", "c": "⚠️", "d": "🔶", "e": "❌"}
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS[grade]}18,{COLORS[grade]}08);
                    border:2px solid {COLORS[grade]}60; border-radius:20px;
                    padding:2.5rem; text-align:center; margin-top:1.5rem;
                    animation:fadeInUp .5s ease;">
            <div style="font-size:5rem;margin-bottom:.5rem">{icons.get(grade,'')}</div>
            <div style="font-size:4rem;font-weight:900;color:{COLORS[grade]};letter-spacing:-2px;
                        line-height:1">{grade.upper()}</div>
            <div style="font-size:1.1rem;color:#444;margin-top:.8rem;font-weight:500">{LABELS[grade]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(["a","b","c","d","e"].index(grade) / 4)

# ── PAGE 5 : Prédiction NLP ──────────────────────────────────────────────
elif page == "🧠 Prédiction NLP (Hugging Face)":
    st.markdown("""
    <div class="page-hdr">
        <h2>🧠 Prédiction NLP — Hugging Face Transformers</h2>
        <p>Modèle : <strong>paraphrase-multilingual-MiniLM-L12-v2</strong> — entrez le nom et les ingrédients, l'IA prédit le Nutri-Score.</p>
    </div>
    """, unsafe_allow_html=True)

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

            icons = {"a": "🌿", "b": "✅", "c": "⚠️", "d": "🔶", "e": "❌"}
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{COLORS[grade]}18,{COLORS[grade]}08);
                        border:2px solid {COLORS[grade]}60; border-radius:20px;
                        padding:2.5rem; text-align:center; margin-top:1.5rem;
                        animation:fadeInUp .5s ease;">
                <div style="font-size:5rem;margin-bottom:.5rem">{icons.get(grade,'')}</div>
                <div style="font-size:4rem;font-weight:900;color:{COLORS[grade]};letter-spacing:-2px;
                            line-height:1">{grade.upper()}</div>
                <div style="font-size:1.1rem;color:#444;margin-top:.8rem;font-weight:500">{LABELS[grade]}</div>
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
    st.markdown("""
    <div class="page-hdr">
        <h2>🔄 Moteur de substitution</h2>
        <p>Choisissez un produit Nutri-Score C/D/E et découvrez des alternatives plus saines dans la même catégorie.</p>
    </div>
    """, unsafe_allow_html=True)

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

                        if st.session_state.user:
                            btn_key = f"save_{i}_{sub_row.get('code','')}"
                            if st.button("💾 Sauvegarder", key=btn_key, use_container_width=True):
                                saved = save_substitute(
                                    user_id        = st.session_state.user["user_id"],
                                    product_code   = str(row.get("code", "")),
                                    product_name   = str(row.get("product_name", "")),
                                    substitute_code= str(sub_row.get("code", "")),
                                    substitute_name= str(sub_row.get("product_name", "")),
                                )
                                if saved:
                                    st.success("Substitut sauvegardé !")
                                else:
                                    st.info("Déjà sauvegardé.")
                        else:
                            st.caption("🔒 Connectez-vous pour sauvegarder")

# ── PAGE 7 : Comparaison ─────────────────────────────────────────────────
elif page == "⚖️ Comparaison produits":
    st.markdown("""
    <div class="page-hdr">
        <h2>⚖️ Comparaison nutritionnelle</h2>
        <p>Sélectionnez deux produits pour comparer leurs valeurs côte à côte avec un radar interactif.</p>
    </div>
    """, unsafe_allow_html=True)

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

# ── PAGE 9 : Connexion / Inscription ────────────────────────────────────────
elif page == "👤 Connexion / Inscription":
    st.markdown("""
    <div class="page-hdr">
        <h2>👤 Connexion / Inscription</h2>
        <p>Créez un compte pour sauvegarder vos substituts préférés et retrouver vos analyses.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user:
        st.success(f"Vous êtes connecté en tant que **{st.session_state.user['username']}**")
        if st.button("Se déconnecter", type="primary"):
            st.session_state.user = None
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["🔑 Connexion", "📝 Inscription"])

        with tab_login:
            st.subheader("Se connecter")
            with st.form("login_form"):
                lu = st.text_input("Nom d'utilisateur")
                lp = st.text_input("Mot de passe", type="password")
                submitted = st.form_submit_button("Se connecter", use_container_width=True)
            if submitted:
                if lu and lp:
                    result = login_user(lu, lp)
                    if result["success"]:
                        st.session_state.user = result
                        st.success(f"Bienvenue, **{result['username']}** !")
                        st.rerun()
                    else:
                        st.error(result["error"])
                else:
                    st.warning("Remplissez tous les champs.")

        with tab_register:
            st.subheader("Créer un compte")
            with st.form("register_form"):
                ru = st.text_input("Nom d'utilisateur")
                re = st.text_input("Email")
                rp = st.text_input("Mot de passe", type="password")
                rp2 = st.text_input("Confirmer le mot de passe", type="password")
                submitted = st.form_submit_button("S'inscrire", use_container_width=True)
            if submitted:
                if not (ru and re and rp):
                    st.warning("Remplissez tous les champs.")
                elif rp != rp2:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    result = register_user(ru, re, rp)
                    if result["success"]:
                        st.session_state.user = result
                        st.success(f"Compte créé ! Bienvenue, **{result['username']}** !")
                        st.rerun()
                    else:
                        st.error(result["error"])

# ── PAGE 10 : Mes substituts ─────────────────────────────────────────────────
elif page == "💾 Mes substituts sauvegardés":
    st.markdown("""
    <div class="page-hdr">
        <h2>💾 Mes substituts sauvegardés</h2>
        <p>Retrouvez toutes vos alternatives nutritionnelles enregistrées.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.user:
        st.warning("Connectez-vous pour accéder à vos substituts.")
        st.stop()

    user_id = st.session_state.user["user_id"]
    subs_df = get_user_substitutes(user_id)

    if subs_df.empty:
        st.info("Vous n'avez pas encore sauvegardé de substituts. Utilisez la page **Substitution produits** pour en ajouter.")
    else:
        st.success(f"**{len(subs_df)}** substitut(s) sauvegardé(s)")
        for _, r in subs_df.iterrows():
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 3, 1])
                with c1:
                    st.markdown(f"**Produit original**")
                    st.markdown(f"🛒 {r.get('product_name', 'N/A')}")
                with c2:
                    st.markdown(f"**Substitut healthier**")
                    st.markdown(f"✅ {r.get('substitute_name', 'N/A')}")
                with c3:
                    st.caption(str(r.get("saved_at", ""))[:10])
                    if st.button("🗑️", key=f"del_{r['id']}", help="Supprimer"):
                        delete_substitute(int(r["id"]), user_id)
                        st.rerun()

# ── PAGE 8 : DuckDB Explorer ─────────────────────────────────────────────
elif page == "🗄️ Explorateur DuckDB":
    st.markdown("""
    <div class="page-hdr">
        <h2>🗄️ Explorateur DuckDB</h2>
        <p>Interrogez directement la base avec du SQL — table <code>products</code> avec toutes les colonnes nutritionnelles.</p>
    </div>
    """, unsafe_allow_html=True)

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

# ── PAGE 11 : Architecture Big Data ──────────────────────────────────────
elif page == "🏗️ Architecture Big Data":
    st.markdown("""
    <div class="page-hdr">
        <h2>🏗️ Architecture Big Data</h2>
        <p>Pipeline complet : ingestion Kafka → stockage DuckDB → traitement Spark → modèles ML/NLP → Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Diagramme architecture ────────────────────────────────────────────
    st.markdown("""
    <style>
    @keyframes flowDown {
        0%  { opacity:.3; transform:translateY(-4px); }
        100%{ opacity:1;  transform:translateY(4px); }
    }
    @keyframes glow {
        0%,100%{ box-shadow: 0 0 0 0 currentColor; }
        50%    { box-shadow: 0 0 18px 4px currentColor; }
    }

    .arch-panel {
        background: linear-gradient(160deg,#0b1e19 0%,#0f2a22 50%,#091a14 100%);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(52,211,153,.15);
        box-shadow: 0 30px 80px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
    }

    /* Layer card */
    .al {
        display: flex;
        align-items: stretch;
        gap: 16px;
        margin-bottom: 6px;
        animation: fadeInUp .5s ease both;
    }
    .al-badge {
        writing-mode: vertical-rl;
        text-orientation: mixed;
        transform: rotate(180deg);
        font-size: .6rem; font-weight: 800; letter-spacing: 2px;
        text-transform: uppercase; padding: 12px 8px;
        border-radius: 10px; min-width: 32px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; white-space: nowrap;
    }
    .al-body {
        flex:1; border-radius: 14px;
        padding: 14px 18px;
        display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
        backdrop-filter: blur(4px);
    }
    .al-title {
        font-size: .7rem; font-weight: 800; letter-spacing: 1.8px;
        text-transform: uppercase; margin-bottom: 10px;
        display: flex; align-items: center; gap: 6px;
        flex-basis: 100%;
    }

    /* Component chip */
    .ac {
        display: flex; align-items: center; gap: 8px;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 12px; padding: 10px 14px;
        transition: all .22s ease; cursor: default;
        min-width: 140px;
    }
    .ac:hover {
        background: rgba(255,255,255,.13);
        border-color: rgba(255,255,255,.22);
        transform: translateY(-3px);
    }
    .ac-icon { font-size: 1.5rem; flex-shrink:0; }
    .ac-info {}
    .ac-name { font-size: .82rem; font-weight: 700; color: #f0fdf9; line-height:1.2; }
    .ac-desc { font-size: .68rem; color: rgba(200,230,220,.6); margin-top:2px; line-height:1.3; }

    /* Arrow connector */
    .ac-arr {
        color: rgba(255,255,255,.25); font-size: 1.1rem;
        display:flex; align-items:center; flex-shrink:0;
    }

    /* Flow arrow between layers */
    .arch-flow {
        display: flex; justify-content: center; align-items: center;
        height: 28px; position: relative; margin: 0;
    }
    .arch-flow::before {
        content: ''; position:absolute;
        width: 2px; height: 100%;
        background: linear-gradient(to bottom, transparent, rgba(52,211,153,.5), transparent);
    }
    .arch-flow-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 10px rgba(52,211,153,.8);
        animation: flowDown .9s ease-in-out infinite alternate;
        position: relative; z-index:1;
    }

    /* Color variants */
    .al-src  .al-badge { background:rgba(251,146,60,.18); color:#fb923c; }
    .al-src  .al-body  { background:rgba(251,146,60,.06); border:1px solid rgba(251,146,60,.18); }
    .al-src  .al-title { color:#fb923c; }

    .al-kafka .al-badge { background:rgba(59,130,246,.18); color:#60a5fa; }
    .al-kafka .al-body  { background:rgba(59,130,246,.06); border:1px solid rgba(59,130,246,.18); }
    .al-kafka .al-title { color:#60a5fa; }

    .al-spark .al-badge { background:rgba(167,139,250,.18); color:#a78bfa; }
    .al-spark .al-body  { background:rgba(167,139,250,.06); border:1px solid rgba(167,139,250,.18); }
    .al-spark .al-title { color:#a78bfa; }

    .al-db .al-badge { background:rgba(45,212,191,.18); color:#2dd4bf; }
    .al-db .al-body  { background:rgba(45,212,191,.06); border:1px solid rgba(45,212,191,.18); }
    .al-db .al-title { color:#2dd4bf; }

    .al-ml .al-badge { background:rgba(52,211,153,.18); color:#34d399; }
    .al-ml .al-body  { background:rgba(52,211,153,.06); border:1px solid rgba(52,211,153,.18); }
    .al-ml .al-title { color:#34d399; }

    .al-app .al-badge { background:rgba(251,191,36,.18); color:#fbbf24; }
    .al-app .al-body  { background:rgba(251,191,36,.06); border:1px solid rgba(251,191,36,.18); }
    .al-app .al-title { color:#fbbf24; }
    </style>

    <div class="arch-panel">

      <!-- SOURCE -->
      <div class="al al-src">
        <div class="al-badge">Source</div>
        <div class="al-body">
          <div class="al-title">📦 Source de données</div>
          <div class="ac">
            <div class="ac-icon">📦</div>
            <div class="ac-info">
              <div class="ac-name">OpenFoodFacts Dump</div>
              <div class="ac-desc">TSV 2.3 GB · ~3M produits</div>
            </div>
          </div>
          <div class="ac-arr">→</div>
          <div class="ac">
            <div class="ac-icon">🌐</div>
            <div class="ac-info">
              <div class="ac-name">API REST OFF</div>
              <div class="ac-desc">Recherche EAN · JSON</div>
            </div>
          </div>
        </div>
      </div>

      <div class="arch-flow"><div class="arch-flow-dot"></div></div>

      <!-- KAFKA -->
      <div class="al al-kafka">
        <div class="al-badge">Streaming</div>
        <div class="al-body">
          <div class="al-title">🔀 Apache Kafka — KRaft mode · Docker port 9092</div>
          <div class="ac">
            <div class="ac-icon">📤</div>
            <div class="ac-info">
              <div class="ac-name">Kafka Producer</div>
              <div class="ac-desc">kafka_producer.py</div>
            </div>
          </div>
          <div class="ac-arr">→</div>
          <div class="ac">
            <div class="ac-icon">🔀</div>
            <div class="ac-info">
              <div class="ac-name">Topic off-products</div>
              <div class="ac-desc">Messages JSON</div>
            </div>
          </div>
          <div class="ac-arr">→</div>
          <div class="ac">
            <div class="ac-icon">📥</div>
            <div class="ac-info">
              <div class="ac-name">Kafka Consumer</div>
              <div class="ac-desc">Batch 500 msgs/s</div>
            </div>
          </div>
        </div>
      </div>

      <div class="arch-flow"><div class="arch-flow-dot"></div></div>

      <!-- SPARK -->
      <div class="al al-spark">
        <div class="al-badge">Traitement</div>
        <div class="al-body">
          <div class="al-title">⚡ Apache Spark — Docker spark-master:7077</div>
          <div class="ac">
            <div class="ac-icon">⚡</div>
            <div class="ac-info">
              <div class="ac-name">Spark Master</div>
              <div class="ac-desc">UI port 8081</div>
            </div>
          </div>
          <div class="ac-arr">→</div>
          <div class="ac">
            <div class="ac-icon">🔧</div>
            <div class="ac-info">
              <div class="ac-name">PySpark Pipeline</div>
              <div class="ac-desc">300k lignes · nettoyage · p99</div>
            </div>
          </div>
          <div class="ac-arr">→</div>
          <div class="ac">
            <div class="ac-icon">🗂️</div>
            <div class="ac-info">
              <div class="ac-name">Parquet Output</div>
              <div class="ac-desc">Columnar format</div>
            </div>
          </div>
        </div>
      </div>

      <div class="arch-flow"><div class="arch-flow-dot"></div></div>

      <!-- DUCKDB -->
      <div class="al al-db">
        <div class="al-badge">Stockage</div>
        <div class="al-body">
          <div class="al-title">🦆 DuckDB — sogood.duckdb</div>
          <div class="ac">
            <div class="ac-icon">🛒</div>
            <div class="ac-info">
              <div class="ac-name">Table products</div>
              <div class="ac-desc">52k produits nettoyés</div>
            </div>
          </div>
          <div class="ac-arr">+</div>
          <div class="ac">
            <div class="ac-icon">👤</div>
            <div class="ac-info">
              <div class="ac-name">Table users</div>
              <div class="ac-desc">Auth SHA-256 + sel</div>
            </div>
          </div>
          <div class="ac-arr">+</div>
          <div class="ac">
            <div class="ac-icon">💾</div>
            <div class="ac-info">
              <div class="ac-name">user_substitutes</div>
              <div class="ac-desc">Favoris utilisateurs</div>
            </div>
          </div>
        </div>
      </div>

      <div class="arch-flow"><div class="arch-flow-dot"></div></div>

      <!-- ML -->
      <div class="al al-ml">
        <div class="al-badge">ML / IA</div>
        <div class="al-body">
          <div class="al-title">🤖 Machine Learning & NLP</div>
          <div class="ac">
            <div class="ac-icon">🤖</div>
            <div class="ac-info">
              <div class="ac-name">XGBoost</div>
              <div class="ac-desc">93.6% acc · 9 features</div>
            </div>
          </div>
          <div class="ac-arr">+</div>
          <div class="ac">
            <div class="ac-icon">🧠</div>
            <div class="ac-info">
              <div class="ac-name">NLP Hugging Face</div>
              <div class="ac-desc">MiniLM · 384-dim</div>
            </div>
          </div>
          <div class="ac-arr">+</div>
          <div class="ac">
            <div class="ac-icon">📈</div>
            <div class="ac-info">
              <div class="ac-name">MLflow</div>
              <div class="ac-desc">Experiment tracking</div>
            </div>
          </div>
        </div>
      </div>

      <div class="arch-flow"><div class="arch-flow-dot"></div></div>

      <!-- APP -->
      <div class="al al-app">
        <div class="al-badge">App</div>
        <div class="al-body">
          <div class="al-title">🥦 SoGood — Streamlit · port 8501</div>
          <div class="ac">
            <div class="ac-icon">🏠</div>
            <div class="ac-info"><div class="ac-name">Dashboard</div><div class="ac-desc">KPIs · Charts</div></div>
          </div>
          <div class="ac">
            <div class="ac-icon">🔍</div>
            <div class="ac-info"><div class="ac-name">Recherche</div><div class="ac-desc">Texte + EAN</div></div>
          </div>
          <div class="ac">
            <div class="ac-icon">🤖</div>
            <div class="ac-info"><div class="ac-name">Prédiction IA</div><div class="ac-desc">XGBoost + NLP</div></div>
          </div>
          <div class="ac">
            <div class="ac-icon">🔄</div>
            <div class="ac-info"><div class="ac-name">Substitution</div><div class="ac-desc">Alternatives saines</div></div>
          </div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Stats live ────────────────────────────────────────────────────────
    st.markdown('<div class="sect">📊 STATISTIQUES EN DIRECT</div>', unsafe_allow_html=True)

    from src.auth import count_users
    import json as _json

    n_products  = _repo.count()
    n_users     = count_users()
    n_subs_row  = _repo.execute_sql("SELECT COUNT(*) as n FROM user_substitutes")
    n_subs      = int(n_subs_row.iloc[0]["n"]) if not n_subs_row.empty and "n" in n_subs_row.columns else 0

    parquet_path = Path(__file__).parent.parent / "data" / "sogood_spark.parquet"
    parquet_info = f"{sum(1 for _ in parquet_path.glob('*.parquet'))} fichier(s)" if parquet_path.exists() else "Non généré"

    xgb_acc, nlp_acc = "—", "—"
    m_path = MODEL_DIR / "metrics.json"
    if m_path.exists():
        xgb_acc = f"{_json.load(open(m_path))['accuracy']*100:.1f}%"
    nlp_path = MODEL_DIR / "nlp_metrics.json"
    if nlp_path.exists():
        nlp_acc = f"{_json.load(open(nlp_path))['accuracy']*100:.1f}%"

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr)">
        <div class="kpi-card">
            <div class="kpi-icon">🦆</div>
            <div class="kpi-value">{n_products:,}</div>
            <div class="kpi-label">Produits DuckDB</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">👤</div>
            <div class="kpi-value">{n_users}</div>
            <div class="kpi-label">Utilisateurs inscrits</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">💾</div>
            <div class="kpi-value">{n_subs}</div>
            <div class="kpi-label">Substituts sauvegardés</div>
        </div>
    </div>
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr)">
        <div class="kpi-card">
            <div class="kpi-icon">🤖</div>
            <div class="kpi-value">{xgb_acc}</div>
            <div class="kpi-label">Accuracy XGBoost</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value">{nlp_acc}</div>
            <div class="kpi-label">Accuracy NLP</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🗂️</div>
            <div class="kpi-value" style="font-size:1.1rem">{parquet_info}</div>
            <div class="kpi-label">Spark Parquet</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Commandes de lancement ────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="sect">🚀 COMMANDES DE LANCEMENT</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .cmd-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:1rem 0 2rem; }
    .cmd-card {
        background: linear-gradient(160deg,#0b1e19,#0f2a22);
        border-radius:16px; padding:18px 20px;
        border:1px solid rgba(52,211,153,.15);
        box-shadow: 0 8px 30px rgba(0,0,0,.2);
    }
    .cmd-card-title {
        font-size:.72rem; font-weight:800; letter-spacing:1.8px; text-transform:uppercase;
        margin-bottom:12px; display:flex; align-items:center; gap:7px;
    }
    .cmd-block {
        background:rgba(0,0,0,.35); border-radius:10px;
        padding:10px 14px; margin-bottom:8px;
        border:1px solid rgba(255,255,255,.07);
        font-family:'Courier New',monospace; font-size:.75rem;
        color:#a7f3d0; line-height:1.6; word-break:break-all;
    }
    .cmd-block span.cmd-comment { color:#4d8c72; }
    </style>

    <div class="cmd-grid">
      <div class="cmd-card">
        <div class="cmd-card-title" style="color:#60a5fa">🔀 Kafka (Docker)</div>
        <div class="cmd-block"><span class="cmd-comment"># Démarrer le broker</span><br>docker-compose up -d kafka</div>
        <div class="cmd-block"><span class="cmd-comment"># Publier les produits</span><br>python src/kafka_producer.py</div>
        <div class="cmd-block"><span class="cmd-comment"># Consommer → DuckDB</span><br>python src/kafka_consumer.py</div>
      </div>
      <div class="cmd-card">
        <div class="cmd-card-title" style="color:#a78bfa">⚡ Spark (Docker)</div>
        <div class="cmd-block"><span class="cmd-comment"># Démarrer le cluster</span><br>docker-compose up -d spark-master spark-worker</div>
        <div class="cmd-block"><span class="cmd-comment"># Lancer le pipeline</span><br>docker exec sogood-spark-master spark-submit --master spark://spark-master:7077 /opt/spark/work/src/spark_pipeline.py</div>
      </div>
      <div class="cmd-card">
        <div class="cmd-card-title" style="color:#34d399">🤖 ML + App</div>
        <div class="cmd-block"><span class="cmd-comment"># Pipeline ML complet</span><br>python main.py</div>
        <div class="cmd-block"><span class="cmd-comment"># Lancer l'application</span><br>streamlit run app/app.py</div>
        <div class="cmd-block"><span class="cmd-comment"># URL locale</span><br>http://localhost:8501</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
