"""
Chatbot nutritionnel IA — powered by Google Gemini 1.5 Flash (gratuit).
Mode démo disponible sans clé API (réponses basées sur DuckDB + règles).
"""

import duckdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

_DB_PATH = str(DATA_DIR / "sogood.duckdb")

NUTRISCORE_LABELS = {
    "a": "Excellent — Très sain ✅",
    "b": "Bon — Bonne qualité nutritionnelle 👍",
    "c": "Moyen — À consommer avec modération ⚠️",
    "d": "Médiocre — Qualité nutritionnelle faible 👎",
    "e": "Mauvais — À éviter ❌",
}

NUTRISCORE_COLORS = {"a": "🟢", "b": "🟩", "c": "🟡", "d": "🟠", "e": "🔴"}

SYSTEM_PROMPT = """Tu es NutriBot, l'assistant IA nutritionnel de SoGood.

SoGood est une plateforme d'analyse nutritionnelle qui contient une base de données
de plus de 52 000 produits alimentaires issus d'Open Food Facts, enrichie par des
modèles de Machine Learning (XGBoost 93.6% de précision) et NLP (Hugging Face).

Tes capacités :
- Analyser la qualité nutritionnelle d'un produit donné
- Expliquer le système Nutri-Score (A=excellent → E=à éviter)
- Comparer deux produits nutritionnellement
- Recommander des alternatives plus saines
- Donner des conseils nutritionnels pratiques et bienveillants
- Répondre à toutes questions sur l'alimentation, les additifs, les allergènes

Nutri-Score :
  A = Excellent (vert foncé) — très faible en sucres, graisses saturées et sel
  B = Bon (vert clair) — bonne qualité nutritionnelle globale
  C = Moyen (jaune) — à consommer avec modération
  D = Médiocre (orange) — qualité nutritionnelle faible
  E = Mauvais (rouge) — riche en sucres, graisses saturées, sel et/ou additifs

Règles de réponse :
- Réponds TOUJOURS en français, de manière chaleureuse et professionnelle
- Sois concis mais complet (3-6 phrases max par réponse)
- Si des données produits sont fournies dans le contexte, utilise-les précisément
- Donne toujours des chiffres concrets quand tu les as
- Termine par un conseil actionnable ou une recommandation pratique
- N'invente jamais de données si elles ne sont pas dans le contexte
"""

# Réponses génériques par thème pour le mode démo
_DEMO_THEMES = {
    ("nutriscore", "nutri-score", "nutri score", "score"): (
        "Le **Nutri-Score** est un système de notation nutritionnelle européen sur 5 niveaux :\n\n"
        "🟢 **A** — Excellent : très peu de sucres, graisses saturées et sel\n"
        "🟩 **B** — Bon : bonne qualité nutritionnelle globale\n"
        "🟡 **C** — Moyen : à consommer avec modération\n"
        "🟠 **D** — Médiocre : qualité faible, à limiter\n"
        "🔴 **E** — Mauvais : riche en sucres, graisses et/ou additifs\n\n"
        "💡 *Conseil : visez des produits A ou B pour composer une alimentation équilibrée.*"
    ),
    ("additif", "additifs", "e100", "colorant", "conservateur"): (
        "Les **additifs alimentaires** sont des substances ajoutées pour conserver, colorer ou améliorer le goût des aliments. "
        "Ils sont identifiés par un code **E** (ex : E621 = glutamate).\n\n"
        "Les plus problématiques : colorants azotés (E102, E110), nitrites (E250), édulcorants en excès.\n"
        "Les plus inoffensifs : vitamine C (E300), lécithine de tournesol (E322).\n\n"
        "💡 *Conseil : moins il y a d'additifs, mieux c'est. Privilégiez les produits avec une liste d'ingrédients courte.*"
    ),
    ("sucre", "sucres", "glucide", "glucides"): (
        "Les **sucres** sont à surveiller dans l'alimentation. L'OMS recommande moins de **25g de sucre libre par jour**.\n\n"
        "Les sodas, jus de fruits et céréales du petit-déjeuner en sont souvent très riches. "
        "Préférez les produits avec moins de **5g de sucres pour 100g**.\n\n"
        "💡 *Conseil : lisez les étiquettes — le sucre se cache sous de nombreux noms (sirop de glucose, fructose, maltose…).*"
    ),
    ("sel", "sodium", "hypertension"): (
        "Le **sel** (sodium) est un facteur de risque cardiovasculaire. L'OMS recommande moins de **5g de sel par jour**.\n\n"
        "Les aliments les plus salés : charcuteries, fromages, plats préparés, soupes en boîte. "
        "Un produit est considéré très salé au-delà de **1.5g de sel pour 100g**.\n\n"
        "💡 *Conseil : cuisinez maison avec des épices et herbes aromatiques pour réduire naturellement votre apport en sel.*"
    ),
    ("proteine", "protéine", "protéines", "proteines"): (
        "Les **protéines** sont essentielles pour les muscles, le système immunitaire et la satiété. "
        "Besoin quotidien : environ **0.8 à 1.2g par kg de poids corporel**.\n\n"
        "Meilleures sources : œufs, légumineuses (lentilles, pois chiches), viande maigre, poisson, tofu.\n\n"
        "💡 *Conseil : variez vos sources de protéines entre animales et végétales pour un meilleur équilibre nutritionnel.*"
    ),
    ("alternative", "remplacer", "substitut", "plus sain", "meilleur"): (
        "Pour trouver des **alternatives plus saines**, voici la méthode SoGood :\n\n"
        "1. Identifiez la catégorie du produit (céréales, boissons, snacks…)\n"
        "2. Cherchez un produit avec un Nutri-Score supérieur (ex : D → B)\n"
        "3. Vérifiez que les protéines et fibres sont plus élevées, sucres/sel plus faibles\n\n"
        "💡 *Utilisez la page **Substitution produits** de SoGood pour des recommandations automatiques basées sur l'IA !*"
    ),
    ("fibre", "fibres"): (
        "Les **fibres alimentaires** favorisent le transit, la satiété et réduisent le risque de diabète de type 2. "
        "Besoin quotidien : **25 à 30g par jour**.\n\n"
        "Meilleures sources : légumineuses, céréales complètes, fruits et légumes frais.\n"
        "Un produit est riche en fibres à partir de **6g pour 100g**.\n\n"
        "💡 *Conseil : remplacez les céréales raffinées (pain blanc, riz blanc) par leurs versions complètes.*"
    ),
}


def _fetch_products(user_message: str) -> list:
    """Retourne les produits trouvés dans DuckDB pour le message donné."""
    try:
        con = duckdb.connect(_DB_PATH)
        words = [w.strip("?.,!") for w in user_message.lower().split() if len(w) > 3]
        products_found = []
        seen = set()
        for word in words[:4]:
            safe = word.replace("'", "''")
            rows = con.execute(f"""
                SELECT product_name, brands, nutriscore_grade,
                       energy_100g, sugars_100g, salt_100g,
                       fat_100g, proteins_100g, additives_n
                FROM products
                WHERE LOWER(product_name) LIKE '%{safe}%'
                   OR LOWER(brands) LIKE '%{safe}%'
                ORDER BY nutriscore_grade ASC
                LIMIT 3
            """).fetchall()
            for row in rows:
                if row[0] not in seen:
                    seen.add(row[0])
                    products_found.append(row)
        con.close()
        return products_found
    except Exception:
        return []


def _fetch_context(user_message: str) -> str:
    """Requête DuckDB pour injecter des données réelles dans le contexte."""
    try:
        con = duckdb.connect(_DB_PATH)
        total = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        dist  = con.execute(
            "SELECT nutriscore_grade, COUNT(*) as n FROM products "
            "GROUP BY nutriscore_grade ORDER BY nutriscore_grade"
        ).fetchall()
        dist_txt = " | ".join(f"{g.upper()}:{n:,}" for g, n in dist if g)
        con.close()

        products = _fetch_products(user_message)
        ctx = f"BASE SOGOOD : {total:,} produits | Distribution : {dist_txt}\n"
        if products:
            ctx += "\nPRODUITS TROUVÉS DANS LA BASE :\n"
            for row in products:
                g = (row[2] or "?").upper()
                label = NUTRISCORE_LABELS.get(row[2] or "?", "")
                ctx += (
                    f"  • {row[0]} ({row[1] or 'Marque inconnue'}) "
                    f"— Nutri-Score {g} ({label})\n"
                    f"    Énergie: {row[3] or 0:.0f} kcal | "
                    f"Sucres: {row[4] or 0:.1f}g | "
                    f"Sel: {row[5] or 0:.2f}g | "
                    f"Graisses: {row[6] or 0:.1f}g | "
                    f"Protéines: {row[7] or 0:.1f}g | "
                    f"Additifs: {int(row[8] or 0)}\n"
                )
        return ctx
    except Exception:
        return "Base SoGood : données nutritionnelles Open Food Facts (52 000+ produits)."


def chat_demo(messages: list) -> str:
    """
    Mode démo sans clé API — utilise DuckDB + réponses basées sur des règles.
    Retourne une réponse nutritionnelle en français avec les vraies données de la base.
    """
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    msg_lower = last_user_msg.lower()

    # Chercher des produits correspondants dans DuckDB
    products = _fetch_products(last_user_msg)

    # Si des produits sont trouvés, construire une réponse avec leurs données
    if products:
        lines = []
        for row in products[:3]:
            name, brand, grade, energy, sugars, salt, fat, proteins, additives_n = row
            g = (grade or "?").lower()
            icon = NUTRISCORE_COLORS.get(g, "⚪")
            label = NUTRISCORE_LABELS.get(g, "Inconnu")
            lines.append(
                f"{icon} **{name}** ({brand or 'Marque inconnue'}) — Nutri-Score **{g.upper()}** — {label}\n"
                f"   Énergie : **{energy or 0:.0f} kcal** | Sucres : **{sugars or 0:.1f}g** | "
                f"Sel : **{salt or 0:.2f}g** | Graisses : **{fat or 0:.1f}g** | "
                f"Protéines : **{proteins or 0:.1f}g** | Additifs : **{int(additives_n or 0)}**"
            )

        response = "Voici ce que j'ai trouvé dans la base SoGood pour votre question :\n\n"
        response += "\n\n".join(lines)

        # Conseil selon le grade du premier produit
        g0 = (products[0][2] or "e").lower()
        if g0 in ("a", "b"):
            response += "\n\n✅ *Ce produit a une bonne qualité nutritionnelle. Vous pouvez le consommer sans inquiétude !*"
        elif g0 == "c":
            response += "\n\n⚠️ *Ce produit est acceptable en consommation modérée. Pensez à varier votre alimentation.*"
        else:
            response += "\n\n❌ *Ce produit est de qualité nutritionnelle faible. Utilisez la page **Substitution** de SoGood pour trouver une alternative plus saine !*"

        return response

    # Sinon, réponse thématique basée sur les mots-clés
    for keywords, reply in _DEMO_THEMES.items():
        if any(kw in msg_lower for kw in keywords):
            return reply

    # Réponse générique si aucune correspondance
    try:
        con = duckdb.connect(_DB_PATH)
        total = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        best = con.execute(
            "SELECT product_name, brands, nutriscore_grade FROM products "
            "WHERE nutriscore_grade = 'a' ORDER BY RANDOM() LIMIT 3"
        ).fetchall()
        con.close()
        exemples = ", ".join(f"**{r[0]}**" for r in best)
        return (
            f"Bonjour ! Je suis NutriBot, votre assistant nutritionnel connecté à la base SoGood "
            f"({total:,} produits analysés).\n\n"
            f"Je peux vous renseigner sur :\n"
            f"- 🔍 La qualité nutritionnelle d'un produit spécifique\n"
            f"- 📊 Le système Nutri-Score (A→E)\n"
            f"- ⚗️ Les additifs alimentaires\n"
            f"- 🔄 Des alternatives plus saines\n\n"
            f"Quelques produits Nutri-Score A dans notre base : {exemples}\n\n"
            f"💡 *Posez-moi une question sur un produit ou un nutriment !*"
        )
    except Exception:
        return (
            "Bonjour ! Je suis NutriBot en mode démo. "
            "Posez-moi une question sur un produit alimentaire, le Nutri-Score ou les additifs !"
        )


def chat(messages: list, api_key: str) -> str:
    """
    Envoie une conversation à Gemini 1.5 Flash et retourne la réponse.

    messages : liste de dict {"role": "user"|"assistant", "content": str}
    api_key  : clé API Google AI Studio (aistudio.google.com) — gratuite
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    db_context = _fetch_context(last_user_msg)
    system_with_context = SYSTEM_PROMPT + f"\n\nCONTEXTE TEMPS RÉEL :\n{db_context}"

    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_with_context,
            max_output_tokens=600,
            temperature=0.7,
        ),
    )

    return response.text
