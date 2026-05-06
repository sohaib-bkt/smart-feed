"""
Endpoint POST /api/interact

Chaque fois qu'un utilisateur like, skip ou regarde un post,
cette route :
  1. Enregistre l'interaction dans Firestore (collection interactions/)
  2. Recalcule l'embedding utilisateur en temps réel
  3. Met à jour users/{user_id}/embedding dans Firestore
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal
from app.db.firebase import (
    log_interaction,
    get_user_interactions,
    update_user_embedding,
)
from app.services.user_profile import compute_user_embedding
from app.models.embeddings import embed_text

router = APIRouter()


class InteractionIn(BaseModel):
    """Corps de la requête POST /interact."""

    user_id: str
    post_id: str
    post_text: str = Field(
        default="",
        description="Texte du post (pour recalculer l'embedding)",
    )
    action: Literal["like", "skip", "watch_full", "watch_50", "comment"]
    watch_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Durée de visionnage en secondes",
    )


@router.post("/interact", summary="Enregistre une interaction utilisateur")
async def record_interaction(body: InteractionIn):
    """
    Enregistre une interaction et met à jour le profil utilisateur.

    **Firestore écrit :**
    - `interactions/` → nouveau document avec timestamp auto
    - `users/{user_id}/embedding` → vecteur recalculé

    **Exemple de payload :**
    ```json
    {
        "user_id": "alice",
        "post_id": "post_42",
        "post_text": "L'IA révolutionne la médecine",
        "action": "like",
        "watch_time": 0.0
    }
    ```
    """

    # ── 1. Calcul de l'embedding du post interagi ─────────────────────
    # Si le texte est fourni → vecteur sémantique réel
    # Sinon → vecteur nul (neutre)
    post_embedding = embed_text(body.post_text) if body.post_text else [0.0] * 384

    # ── 2. Log de l'interaction dans Firestore ────────────────────────
    interaction_data = {
        "user_id":    body.user_id,
        "post_id":    body.post_id,
        "action":     body.action,
        "watch_time": body.watch_time,
        "embedding":  post_embedding,   # stocké pour compute_user_embedding()
    }
    await log_interaction(interaction_data)
    # → Firestore : ajoute un doc dans interactions/ avec timestamp auto

    # ── 3. Recalcul de l'embedding utilisateur ────────────────────────
    interactions = await get_user_interactions(body.user_id, last_n=100)
    new_embedding = compute_user_embedding(interactions)
    await update_user_embedding(body.user_id, new_embedding)
    # → Firestore : met à jour users/{user_id}/embedding

    return {"status": "logged", "user_id": body.user_id}