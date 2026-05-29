"""
Chatbot nutritionnel IA — powered by Claude API (Anthropic).

Fonctionnement :
  1. La question de l'utilisateur est analysée pour extraire des mots-clés.
  2. Ces mots-clés servent à requêter DuckDB et injecter des données réelles
     dans le contexte envoyé à Claude.
  3. Claude répond en français en s'appuyant sur ces données.
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


def _fetch_context(user_message: str) -> str:
    """Requête DuckDB pour injecter des données réelles dans le contexte."""
    try:
        con = duckdb.connect(_DB_PATH)

        # Stats générales
        total = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        dist  = con.execute(
            "SELECT nutriscore_grade, COUNT(*) as n FROM products "
            "GROUP BY nutriscore_grade ORDER BY nutriscore_grade"
        ).fetchall()
        dist_txt = " | ".join(f"{g.upper()}:{n:,}" for g, n in dist if g)

        # Recherche de produits mentionnés dans le message
        words = [w.strip("?.,!") for w in user_message.lower().split() if len(w) > 3]
        products_found = []
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
            products_found.extend(rows)

        con.close()

        ctx = f"BASE SOGOOD : {total:,} produits | Distribution : {dist_txt}\n"

        if products_found:
            ctx += "\nPRODUITS TROUVÉS DANS LA BASE :\n"
            seen = set()
            for row in products_found:
                name = row[0]
                if name in seen:
                    continue
                seen.add(name)
                g = (row[2] or "?").upper()
                label = NUTRISCORE_LABELS.get(row[2] or "?", "")
                ctx += (
                    f"  • {name} ({row[1] or 'Marque inconnue'}) "
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


def chat(messages: list, api_key: str) -> str:
    """
    Envoie une conversation à Claude et retourne la réponse.

    messages : liste de dict {"role": "user"|"assistant", "content": str}
    api_key  : clé API Anthropic
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Injecter le contexte DuckDB basé sur le dernier message utilisateur
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    db_context = _fetch_context(last_user_msg)

    system_with_context = SYSTEM_PROMPT + f"\n\nCONTEXTE TEMPS RÉEL :\n{db_context}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_with_context,
        messages=messages,
    )

    return response.content[0].text
