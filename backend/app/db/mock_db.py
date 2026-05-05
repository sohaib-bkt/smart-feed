"""
Version mock de firebase.py pour les tests — aucune connexion réseau nécessaire.
Utilise de simples dictionnaires Python à la place de Firestore.

Usage dans les tests :
    from app.db.mock_db import get_user_profile, create_user, ...
    (à la place de : from app.db.firebase import ...)

Pour reset l'état entre les tests, appeler : reset_mock()
"""

from datetime import datetime
import copy

# ── Stockage en mémoire (simule les collections Firestore) ────────────
_users_store: dict[str, dict] = {}
_interactions_store: list[dict] = []


def reset_mock() -> None:
    """Vide toutes les données — à appeler dans les fixtures pytest."""
    global _users_store, _interactions_store
    _users_store.clear()
    _interactions_store.clear()


# ── USERS ─────────────────────────────────────────────────────────────

async def get_user_profile(user_id: str) -> dict | None:
    """Retourne le profil ou None si inexistant."""
    user = _users_store.get(user_id)
    return copy.deepcopy(user) if user else None


async def create_user(user_id: str, preferences: dict = None) -> dict:
    """Crée un utilisateur avec valeurs par défaut."""
    profile = {
        "user_id": user_id,
        "preferences": preferences or {
            "toxicity_threshold": 0.3,
            "interests": [],
            "mode": "default",
            "content_type": "all",
        },
        "embedding": [0.0] * 384,
        "created_at": datetime.now().isoformat(),
    }
    _users_store[user_id] = copy.deepcopy(profile)
    return profile


async def update_user_preferences(user_id: str, prefs: dict) -> None:
    """Met à jour les préférences. Lève une erreur si l'user n'existe pas."""
    if user_id not in _users_store:
        raise KeyError(f"Utilisateur '{user_id}' introuvable")
    _users_store[user_id]["preferences"] = copy.deepcopy(prefs)


async def update_user_embedding(user_id: str, embedding: list) -> None:
    """Met à jour le vecteur embedding. Lève une erreur si l'user n'existe pas."""
    if user_id not in _users_store:
        raise KeyError(f"Utilisateur '{user_id}' introuvable")
    _users_store[user_id]["embedding"] = list(embedding)


# ── INTERACTIONS ──────────────────────────────────────────────────────

async def log_interaction(data: dict) -> None:
    """Enregistre une interaction avec timestamp automatique."""
    entry = copy.deepcopy(data)
    entry["timestamp"] = datetime.now().isoformat()
    _interactions_store.append(entry)


async def get_user_interactions(user_id: str, last_n: int = 100) -> list:
    """
    Retourne les N dernières interactions de l'utilisateur,
    triées du plus récent au plus ancien (comme Firestore).
    """
    user_interactions = [
        i for i in _interactions_store if i.get("user_id") == user_id
    ]
    # Tri décroissant par timestamp (string ISO → ordre alphabétique = chronologique)
    user_interactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return copy.deepcopy(user_interactions[:last_n])