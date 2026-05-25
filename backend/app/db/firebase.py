"""
Couche d'accès à Firestore (base de données Firebase).

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

def _resolve_cred_path() -> Path:
    """Resolve credentials path, trying configured path and common alternatives."""
    cred_path = Path(settings.firebase_credentials)
    if not cred_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        cred_path = backend_root / settings.firebase_credentials
    
    if cred_path.exists():
        return cred_path
    
    # Fallback: try common alternative paths
    alt_names = ["servicesAccountKey.json", "serviceAccountKey.json"]
    backend_root = Path(__file__).resolve().parents[2]
    for name in alt_names:
        alt = backend_root / name
        if alt.exists():
            return alt
    
    return cred_path


def _init_firebase():
    """Initialise Firebase Admin SDK une seule fois."""
    global _db
    if _db is not None:
        return _db
    
    cred_path = _resolve_cred_path()
    
    if not cred_path.exists():
        raise FileNotFoundError(
            f"Firebase credentials not found at {cred_path}. "
            f"Please ensure serviceAccountKey.json is in the backend directory."
        )
    
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
    """
    Récupère le profil d'un utilisateur depuis Firestore.
    Retourne None si l'utilisateur n'existe pas encore.
    """
    db = get_db()
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


async def create_user(user_id: str, preferences: dict = None) -> dict:
    """
    Crée un nouveau profil utilisateur avec des valeurs par défaut.
    Si le profil existe déjà, retourne l'existant (sans écraser).
    """
    db = get_db()
    doc_ref = db.collection("users").document(user_id)
    doc = doc_ref.get()
    
    if doc.exists:
        # Retourner le profil existant sans modification
        return doc.to_dict()
    
    # Créer nouveau profil
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
    """Met à jour uniquement les préférences d'un utilisateur."""
    db = get_db()
    db.collection("users").document(user_id).update({"preferences": prefs})


async def update_user_embedding(user_id: str, embedding: list) -> None:
    """Met à jour le vecteur d'intérêts de l'utilisateur."""
    db = get_db()
    db.collection("users").document(user_id).update({"embedding": embedding})


async def delete_user(user_id: str) -> bool:
    """Supprime un utilisateur et toutes ses interactions."""
    db = get_db()
    
    # Supprimer le profil
    db.collection("users").document(user_id).delete()
    
    # Supprimer toutes ses interactions
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
    """
    Enregistre une interaction utilisateur.
    Retourne l'ID du document créé.
    """
    db = get_db()
    data["timestamp"] = datetime.now().isoformat()
    doc_ref = db.collection("interactions").add(data)
    return doc_ref[1].id


async def get_all_interactions(limit: int = None) -> list:
    """
    Récupère toutes les interactions de Firestore.
    Optionnellement avec une limite.
    """
    db = get_db()
    query = db.collection("interactions")
    if limit:
        query = query.limit(limit)
    docs = query.stream()
    return [d.to_dict() for d in docs]


async def get_user_interactions(user_id: str, last_n: int = 100) -> list:
    """
    Récupère les N dernières interactions d'un utilisateur,
    triées du plus récent au plus ancien.
    """
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
    """
    Récupère les interactions d'un utilisateur pour un type d'action spécifique.
    Exemple: likes, skips, etc.
    """
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
    """Compte le nombre total d'interactions d'un utilisateur."""
    db = get_db()
    docs = db.collection("interactions").where("user_id", "==", user_id).stream()
    return sum(1 for _ in docs)


async def delete_all_interactions() -> int:
    """Supprime TOUTES les interactions (⚠️ Attention!). Retourne le nombre supprimé."""
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
    """Vérifie que Firebase est accessible."""
    try:
        db = get_db()
        # Tenter une opération simple
        db.collection("users").limit(1).stream()
        return True
    except Exception as e:
        print(f"Firebase connection error: {e}")
        return False


def is_available() -> bool:
    """Retourne True si Firebase est configuré et accessible."""
    try:
        return _resolve_cred_path().exists()
    except Exception:
        return False