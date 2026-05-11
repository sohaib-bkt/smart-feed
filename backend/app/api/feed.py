"""
Endpoint GET /api/feed/{user_id}

Pipeline complet :
    1. Charger (ou créer) le profil utilisateur depuis Firestore
    2. Recalculer son embedding depuis ses 100 dernières interactions
    3. Détecter ses intérêts si non renseignés
    4. Appeler le moteur de recommandation (FAISS + scoring)
    5. Retourner le feed trié
"""

from fastapi import APIRouter, HTTPException, Query
from app.services.recommender import get_feed, get_feed_v2
from app.services.user_profile import compute_user_embedding, get_top_interests
from app.db.firebase import get_user_profile, get_user_interactions, create_user

router = APIRouter()


@router.get("/feed/{user_id}", summary="Feed personnalisé d'un utilisateur")
async def get_user_feed(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=50, description="Nombre de posts (max 50)"),
    version: str = Query(default="v1", pattern="^v[12]$"),
):
    """
    Retourne le feed personnalisé pour un utilisateur.

    - Si l'utilisateur n'existe pas encore → profil créé automatiquement
    - L'embedding est recalculé à chaque appel depuis l'historique réel
    - Les intérêts sont auto-détectés si non définis manuellement

    **Firestore écrit/lit :**
    - `users/{user_id}` → profil + préférences
    - `interactions/{user_id}` → historique pour recalculer l'embedding
    """

    # ── 1. Charger ou créer le profil ──────────────────────────────────
    profile = await get_user_profile(user_id)
    if not profile:
        profile = await create_user(user_id)
        # → Firestore : crée le document users/{user_id}

    # ── 2. Recalculer l'embedding depuis l'historique ──────────────────
    interactions = await get_user_interactions(user_id, last_n=100)
    # → Firestore : lit la collection interactions/ filtrée par user_id
    user_emb = compute_user_embedding(interactions)

    # ── 3. Enrichir les intérêts si vides ─────────────────────────────
    prefs = profile.get("preferences", {})
    if not prefs.get("interests") and interactions:
        prefs["interests"] = get_top_interests(interactions)

    # ── 4. Générer le feed ─────────────────────────────────────────────
    if version == "v2":
        feed_items = get_feed_v2(
            user_embedding=user_emb,
            user_prefs=prefs,
            n_results=limit,
        )
    else:
        feed_items = get_feed(
            user_embedding=user_emb,
            user_prefs=prefs,
            n_results=limit,
        )

    return {
        "user_id": user_id,
        "version": version,
        "mode": prefs.get("mode", "default"),
        "count": len(feed_items),
        "feed": feed_items,
    }