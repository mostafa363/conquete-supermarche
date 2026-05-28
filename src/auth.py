"""
Couche authentification — gestion des utilisateurs et des favoris.

Stockage dans DuckDB (tables users + user_substitutes).
Mots de passe hashés avec SHA-256 + sel aléatoire.
"""

import duckdb
import hashlib
import secrets
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

_DB_PATH = str(DATA_DIR / "sogood.duckdb")


# ── Connexion ────────────────────────────────────────────────────────────────

def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(_DB_PATH)


# ── Initialisation des tables ────────────────────────────────────────────────

def init_auth_tables():
    """Crée les tables users et user_substitutes si elles n'existent pas."""
    con = _connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER,
            username      VARCHAR NOT NULL,
            email         VARCHAR NOT NULL,
            password_hash VARCHAR NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_substitutes (
            id              INTEGER,
            user_id         INTEGER NOT NULL,
            product_code    VARCHAR,
            product_name    VARCHAR,
            substitute_code VARCHAR,
            substitute_name VARCHAR,
            saved_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.close()


# ── Hachage mots de passe ────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == h
    except Exception:
        return False


# ── Inscription ──────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> dict:
    username = username.strip()
    email    = email.strip().lower()

    if not username or not email or not password:
        return {"success": False, "error": "Tous les champs sont obligatoires."}
    if len(password) < 6:
        return {"success": False, "error": "Mot de passe trop court (6 caractères min)."}

    con = _connect()
    try:
        existing = con.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            [username, email]
        ).fetchone()
        if existing:
            return {"success": False, "error": "Nom d'utilisateur ou email déjà utilisé."}

        row = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM users").fetchone()
        new_id = row[0]
        ph = _hash_password(password)
        con.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            [new_id, username, email, ph]
        )
        return {"success": True, "user_id": new_id, "username": username}
    finally:
        con.close()


# ── Connexion ────────────────────────────────────────────────────────────────

def login_user(username: str, password: str) -> dict:
    con = _connect()
    try:
        row = con.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            [username.strip()]
        ).fetchone()
        if not row:
            return {"success": False, "error": "Utilisateur introuvable."}
        user_id, uname, ph = row
        if not _verify_password(password, ph):
            return {"success": False, "error": "Mot de passe incorrect."}
        return {"success": True, "user_id": user_id, "username": uname}
    finally:
        con.close()


# ── Substituts ───────────────────────────────────────────────────────────────

def save_substitute(user_id: int, product_code: str, product_name: str,
                    substitute_code: str, substitute_name: str) -> bool:
    con = _connect()
    try:
        existing = con.execute(
            """SELECT id FROM user_substitutes
               WHERE user_id = ? AND product_code = ? AND substitute_code = ?""",
            [user_id, str(product_code), str(substitute_code)]
        ).fetchone()
        if existing:
            return False

        row = con.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM user_substitutes"
        ).fetchone()
        new_id = row[0]
        con.execute(
            """INSERT INTO user_substitutes
               (id, user_id, product_code, product_name, substitute_code, substitute_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [new_id, user_id, str(product_code), str(product_name),
             str(substitute_code), str(substitute_name)]
        )
        return True
    finally:
        con.close()


def get_user_substitutes(user_id: int) -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute(
            "SELECT * FROM user_substitutes WHERE user_id = ? ORDER BY saved_at DESC",
            [user_id]
        ).df()
    finally:
        con.close()


def delete_substitute(sub_id: int, user_id: int) -> bool:
    con = _connect()
    try:
        con.execute(
            "DELETE FROM user_substitutes WHERE id = ? AND user_id = ?",
            [sub_id, user_id]
        )
        return True
    finally:
        con.close()


def count_users() -> int:
    con = _connect()
    try:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    except Exception:
        return 0
    finally:
        con.close()
