"""
Endpoint GET /api/feed/{user_id}

v1 —> content-based pur (FAISS + scoring Phase 1)
v2 —> hybride (FAISS + LightFM) + re-ranking final via rank_candidates

Pipeline v2 :
    1. Charger (ou créer) le profil utilisateur depuis Firestore
    2. Recalculer son embedding depuis ses 100 dernières interactions
    3. Détecter ses intérêts si non renseignés
    4. get_hybrid_feed()  → 60 candidats scorés (content-based + CF)
    5. rank_candidates()  → re-ranking final pondéré (similarité, toxicité,
                            popularité, fraîcheur)
    6. Retourner les limit premiers
"""

from fastapi import APIRouter, Query
from app.services.recommender import get_feed
from app.services.hybrid_recommender import get_hybrid_feed
from app.models.ranker import rank_candidates
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

    - **v1** : content-based pur (Phase 1 — FAISS + scoring pondéré)
    - **v2** : hybride content-based + collaboratif, puis re-ranking final

    Si l'utilisateur n'existe pas encore → profil créé automatiquement.

    **Firestore lit :**
    - `users/{user_id}`-> profil + préférences
    - `interactions/{user_id}` -> historique pour recalculer l'embedding
    """

    # 1. Charger ou créer le profil ──────────────────────────────────
    profile = await get_user_profile(user_id)
    if not profile:
        profile = await create_user(user_id)

    # 2. Recalculer l'embedding depuis l'historique ──────────────────
    interactions = await get_user_interactions(user_id, last_n=100)
    user_emb = compute_user_embedding(interactions)

    # 3. Enrichir les intérêts si vides ─────────────────────────────
    prefs = profile.get("preferences", {})
    if not prefs.get("interests") and interactions:
        prefs["interests"] = get_top_interests(interactions)

    # 4. Générer le feed ─────────────────────────────────────────────
    if version == "v2":
        # Feed hybride : content-based (FAISS) + collaboratif (LightFM)
        # → 60 candidats pré-scorés
        candidates = get_hybrid_feed(
            user_id=user_id,
            user_embedding=user_emb,
            user_prefs=prefs,
            n_candidates=300,
            n_results=60,
        )

        # Re-ranking final : formule pondérée (similarité, toxicité, pop, fraîcheur)
        feed_items = rank_candidates(candidates, user_embedding=user_emb)[:limit]
        ranking_method = "hybrid_xgboost"

    else:
        # Fallback v1 — content-based pur Phase 1
        feed_items = get_feed(
            user_embedding=user_emb,
            user_prefs=prefs,
            n_results=limit,
        )
        ranking_method = "content_based_v1"

    return {
        "user_id": user_id,
        "version": version,
        "ranking_method": ranking_method,
        "mode": prefs.get("mode", "default"),
        "count": len(feed_items),
        "feed": feed_items,
    }