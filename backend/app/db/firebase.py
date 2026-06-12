"""
Couche d'accès à Firestore (base de données Firebase).
Fallback automatique sur mock en mémoire si Firebase n'est pas configuré.

Structure des collections Firestore :
  users/
    {user_id}/
      user_id: str
      preferences: dict
      embedding: list[float]   (384 dimensions)
      created_at: str          (ISO 8601)

  interactions/
    {auto_id}/
      user_id: str
      post_id: str
      action: str            (like / skip / watch_full / ...)
      timestamp: str         (ISO 8601)
"""

import os
from pathlib import Path
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

from app.config import settings

# ─────────────────────────────────────────────────────────────────────
# Initialisation Firebase (Singleton)
# ─────────────────────────────────────────────────────────────────────

_db = None
_use_mock = False


def _resolve_cred_path() -> Path:
    """Resolve credentials path, trying configured path and common alternatives."""
    cred_path = Path(settings.firebase_credentials)
    if not cred_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        cred_path = backend_root / settings.firebase_credentials

    if cred_path.exists():
        return cred_path

    alt_names = ["servicesAccountKey.json", "serviceAccountKey.json"]
    backend_root = Path(__file__).resolve().parents[2]
    for name in alt_names:
        alt = backend_root / name
        if alt.exists():
            return alt

    return cred_path


def _init_firebase():
    """Initialise Firebase Admin SDK une seule fois. Fallback sur mock si absent."""
    global _db, _use_mock
    if _db is not None:
        return _db

    cred_path = _resolve_cred_path()

    if not cred_path.exists():
        print("ℹ️  Firebase credentials not found — using in-memory mock DB")
        _use_mock = True
        return None

    cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    print(f"✅ Firebase initialisé avec: {cred_path}")
    return _db


def get_db():
    """Retourne le client Firestore (initialisé au premier appel)."""
    global _db
    if _db is None:
        _init_firebase()
    return _db


# ─────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────

async def get_user_profile(user_id: str) -> dict | None:
    if _use_mock:
        return await _mock_get_user_profile(user_id)
    db = get_db()
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


async def create_user(user_id: str, preferences: dict = None) -> dict:
    if _use_mock:
        return await _mock_create_user(user_id, preferences)
    db = get_db()
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
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
    doc_ref.set(profile)
    return profile


async def update_user_preferences(user_id: str, prefs: dict) -> None:
    if _use_mock:
        return await _mock_update_user_preferences(user_id, prefs)
    db = get_db()
    db.collection("users").document(user_id).update({"preferences": prefs})


async def update_user_embedding(user_id: str, embedding: list) -> None:
    if _use_mock:
        return await _mock_update_user_embedding(user_id, embedding)
    db = get_db()
    db.collection("users").document(user_id).update({"embedding": embedding})


async def delete_user(user_id: str) -> bool:
    if _use_mock:
        return _mock_delete_user(user_id)
    db = get_db()
    db.collection("users").document(user_id).delete()
    interactions = db.collection("interactions").where("user_id", "==", user_id).stream()
    batch = db.batch()
    for doc in interactions:
        batch.delete(doc.reference)
    batch.commit()
    return True


# ─────────────────────────────────────────────────────────────────────
# INTERACTIONS
# ─────────────────────────────────────────────────────────────────────

async def log_interaction(data: dict) -> str:
    if _use_mock:
        return _mock_log_interaction(data)
    db = get_db()
    data["timestamp"] = datetime.now().isoformat()
    doc_ref = db.collection("interactions").add(data)
    return doc_ref[1].id


async def get_all_interactions(limit: int = None) -> list:
    if _use_mock:
        return _mock_get_all_interactions()
    db = get_db()
    query = db.collection("interactions")
    if limit:
        query = query.limit(limit)
    docs = query.stream()
    return [d.to_dict() for d in docs]


async def get_user_interactions(user_id: str, last_n: int = 100) -> list:
    if _use_mock:
        return _mock_get_user_interactions(user_id, last_n)
    db = get_db()
    docs = (
        db.collection("interactions")
        .where("user_id", "==", user_id)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(last_n)
        .stream()
    )
    return [d.to_dict() for d in docs]


async def get_interactions_by_action(user_id: str, action: str, limit: int = 50) -> list:
    if _use_mock:
        return _mock_get_interactions_by_action(user_id, action, limit)
    db = get_db()
    docs = (
        db.collection("interactions")
        .where("user_id", "==", user_id)
        .where("action", "==", action)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [d.to_dict() for d in docs]


async def count_user_interactions(user_id: str) -> int:
    if _use_mock:
        return _mock_count_user_interactions(user_id)
    db = get_db()
    docs = db.collection("interactions").where("user_id", "==", user_id).stream()
    return sum(1 for _ in docs)


async def delete_all_interactions() -> int:
    if _use_mock:
        return _mock_delete_all_interactions()
    db = get_db()
    docs = db.collection("interactions").stream()
    count = 0
    batch = db.batch()
    for doc in docs:
        batch.delete(doc.reference)
        count += 1
        if count % 500 == 0:
            batch.commit()
            batch = db.batch()
    if count % 500 != 0:
        batch.commit()
    return count


# ─────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────

async def check_connection() -> bool:
    if _use_mock:
        return True
    try:
        db = get_db()
        db.collection("users").limit(1).stream()
        return True
    except Exception as e:
        print(f"Firebase connection error: {e}")
        return False


def is_available() -> bool:
    if _use_mock:
        return True
    try:
        return _resolve_cred_path().exists()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# MOCK DB — fallback en mémoire (sans Firebase)
# ─────────────────────────────────────────────────────────────────────

_mock_users: dict[str, dict] = {}
_mock_interactions: list[dict] = []


async def _mock_get_user_profile(user_id: str) -> dict | None:
    u = _mock_users.get(user_id)
    return dict(u) if u else None


async def _mock_create_user(user_id: str, preferences: dict | None = None) -> dict:
    if user_id in _mock_users:
        return dict(_mock_users[user_id])
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
    _mock_users[user_id] = dict(profile)
    return profile


async def _mock_update_user_preferences(user_id: str, prefs: dict) -> None:
    if user_id not in _mock_users:
        _mock_users[user_id] = {"preferences": {}}
    _mock_users[user_id]["preferences"] = dict(prefs)


async def _mock_update_user_embedding(user_id: str, embedding: list) -> None:
    if user_id in _mock_users:
        _mock_users[user_id]["embedding"] = list(embedding)


def _mock_delete_user(user_id: str) -> bool:
    _mock_users.pop(user_id, None)
    _mock_interactions[:] = [i for i in _mock_interactions if i.get("user_id") != user_id]
    return True


def _mock_log_interaction(data: dict) -> str:
    entry = dict(data)
    entry["timestamp"] = datetime.now().isoformat()
    _mock_interactions.append(entry)
    return str(id(entry))


def _mock_get_all_interactions() -> list:
    return list(_mock_interactions)


def _mock_get_user_interactions(user_id: str, last_n: int = 100) -> list:
    user_ints = [i for i in _mock_interactions if i.get("user_id") == user_id]
    user_ints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return user_ints[:last_n]


def _mock_get_interactions_by_action(user_id: str, action: str, limit: int = 50) -> list:
    user_ints = [i for i in _mock_interactions if i.get("user_id") == user_id and i.get("action") == action]
    user_ints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return user_ints[:limit]


def _mock_count_user_interactions(user_id: str) -> int:
    return sum(1 for i in _mock_interactions if i.get("user_id") == user_id)


def _mock_delete_all_interactions() -> int:
    count = len(_mock_interactions)
    _mock_interactions.clear()
    return count
