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
from app.config import settings

def _credentials_missing() -> bool:
    cred_path = Path(settings.firebase_credentials)
    if cred_path.is_absolute():
        return not cred_path.exists()
    backend_root = Path(__file__).resolve().parents[2]
    return not (backend_root / cred_path).exists()


_USE_MOCK = settings.environment == "development" and _credentials_missing()

if _USE_MOCK:
    from app.db import mock_db as _mock_db

    def get_db():
        return None

    async def get_all_interactions() -> list:
        return await _mock_db.get_all_interactions()

    get_user_profile = _mock_db.get_user_profile
    create_user = _mock_db.create_user
    update_user_preferences = _mock_db.update_user_preferences
    update_user_embedding = _mock_db.update_user_embedding
    log_interaction = _mock_db.log_interaction
    get_user_interactions = _mock_db.get_user_interactions

else:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from datetime import datetime

# Singleton : une seule connexion partagée ──────────────────────────
    _db = None


    def get_db():
        """
        Retourne le client Firestore.
        Initialise Firebase au premier appel uniquement (pattern singleton).
        """
        global _db
        if _db is None:
            # Charge le fichier serviceAccountKey.json (clé secrète)
            cred = credentials.Certificate(settings.firebase_credentials)
            firebase_admin.initialize_app(cred)
            _db = firestore.client()
        return _db


# ── USERS ─────────────────────────────────────────────────────────────

    async def get_user_profile(user_id: str) -> dict | None:
        """
        Récupère le profil d'un utilisateur depuis Firestore.
        Retourne None si l'utilisateur n'existe pas encore.
        """
        db = get_db()
        doc = db.collection("users").document(user_id).get()
        return doc.to_dict() if doc.exists else None


    async def create_user(user_id: str, preferences: dict = None) -> dict:
        """
        Crée un nouveau profil utilisateur avec des valeurs par défaut.
        Si le profil existe déjà, il sera écrasé (idempotent).
        """
        db = get_db()
        profile = {
            "user_id": user_id,
            "preferences": preferences or {
                "toxicity_threshold": 0.3,
                "interests": [],
                "mode": "default",
                "content_type": "all",
            },
            "embedding": [0.0] * 384,  # vecteur neutre au départ
            "created_at": datetime.now().isoformat(),
        }
        db.collection("users").document(user_id).set(profile)
        return profile


    async def update_user_preferences(user_id: str, prefs: dict) -> None:
        """
        Met à jour uniquement les préférences d'un utilisateur.
        Les autres champs (embedding, created_at) ne sont pas touchés.
        """
        db = get_db()
        db.collection("users").document(user_id).update({"preferences": prefs})


    async def update_user_embedding(user_id: str, embedding: list) -> None:
        """
        Met à jour le vecteur d'intérêts de l'utilisateur.
        Appelé après chaque nouvelle interaction pour affiner le feed.
        """
        db = get_db()
        db.collection("users").document(user_id).update({"embedding": embedding})


# ── INTERACTIONS ──────────────────────────────────────────────────────

    async def log_interaction(data: dict) -> None:
        """
        Enregistre une interaction utilisateur (like, skip, watch...).
        Firestore génère automatiquement un ID unique pour le document.

        Exemple de data attendu :
            {
                "user_id": "alice",
                "post_id": "post_42",
                "action": "like"
            }
        """
        db = get_db()
        data["timestamp"] = datetime.now().isoformat()
        db.collection("interactions").add(data)  # .add() = ID auto-généré


    async def get_all_interactions() -> list:
        """
        Récupère TOUTES les interactions de la base Firestore.
        Used for building the collaborative filtering matrix.
        """
        db = get_db()
        docs = db.collection("interactions").stream()
        return [d.to_dict() for d in docs]


    async def get_user_interactions(user_id: str, last_n: int = 100) -> list:
        """
        Récupère les N dernières interactions d'un utilisateur,
        triées du plus récent au plus ancien.
        Utilisé par compute_user_embedding() pour recalculer le vecteur.
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