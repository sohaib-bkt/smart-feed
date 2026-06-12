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
from app.services.recommender import get_feed, get_feed_v2, get_feed_v3
from app.services.recommender import _media_url_for_text_post, _media_url_for_image_post, _media_url_for_video_post
from app.services.hybrid_recommender import get_hybrid_feed
from app.models.ranker import rank_candidates
from app.services.user_profile import compute_user_embedding, get_top_interests
from app.db.firebase import get_user_profile, get_user_interactions, create_user

router = APIRouter()

# Chaque mode de navigation est associé à un ensemble de catégories
MODE_CATEGORY_MAP: dict[str, list[str]] = {
    "default": [],
    "focus":   ["POLITICS", "BUSINESS", "HOME & LIVING", "BLACK VOICES", "QUEER VOICES", "PARENTS"],
    "fun":     ["COMEDY", "ENTERTAINMENT", "SPORTS", "FOOD & DRINK", "TRAVEL", "STYLE & BEAUTY"],
    "learning":["WELLNESS", "HEALTHY LIVING", "PARENTING"],
    "fresh":   [],
}


@router.get("/feed/{user_id}", summary="Feed personnalisé d'un utilisateur")
async def get_user_feed(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=50, description="Nombre de posts (max 50)"),
    version: str = Query(default="v1", pattern="^v[123]$"),
    content_type: str = Query(default=None, description="Filtrer par type : text, image, video"),
    mode: str = Query(default=None, description="Mode de recommandation"),
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

    # Surcharger les préférences avec les paramètres de requête si fournis
    if mode is not None:
        prefs["mode"] = mode
    if content_type is not None:
        prefs["content_type"] = content_type

    # Résoudre les catégories autorisées pour le mode sélectionné
    effective_mode = prefs.get("mode", "default")
    allowed_categories = MODE_CATEGORY_MAP.get(effective_mode, [])

    # 4. Générer le feed ─────────────────────────────────────────────
    if version == "v2":
        candidates = get_hybrid_feed(
            user_id=user_id,
            user_embedding=user_emb,
            user_prefs=prefs,
            n_candidates=300,
            n_results=60,
            diversify=True,
            max_per_category=3
        )
        feed_items = rank_candidates(candidates, user_embedding=user_emb)[:limit]
        ranking_method = "hybrid_xgboost"

    elif version == "v3":
        feed_items = get_feed_v3(
            user_embedding=user_emb,
            user_prefs=prefs,
            n_results=limit,
            embedding_dim=384,
        )
        ranking_method = "content_based_v3"

    else:
        feed_items = get_feed(
            user_embedding=user_emb,
            user_prefs=prefs,
            n_results=limit,
        )
        ranking_method = "content_based_v1"

    # Filtrer par catégories si le mode en définit
    if allowed_categories:
        feed_items = [p for p in feed_items if p.get("category", "").upper() in allowed_categories]

        # Si pas assez de résultats, compléter depuis le dataframe
        if len(feed_items) < limit:
            existing_ids = {p.get("id") for p in feed_items}
            try:
                import pandas as pd
                df = pd.read_parquet("data/processed/combined_meta_v3.parquet")
                cat_df = df[df["category"].str.upper().isin(allowed_categories)]
                cat_df = cat_df[~cat_df.index.isin([int(i) for i in existing_ids if i.isdigit()])]
                needed = limit - len(feed_items)
                if len(cat_df) > 0:
                    sampled = cat_df.sample(n=min(needed, len(cat_df)), random_state=42)
                    for _, row in sampled.iterrows():
                        modal = str(row.get("modal", "text"))
                        text = str(row.get("text", ""))[:200]
                        category = str(row.get("category", ""))
                        if modal == "video":
                            media = _media_url_for_video_post(text)
                            video_url = media.get("video_url")
                            image_url = media.get("image_url")
                        elif modal == "image":
                            image_url = _media_url_for_image_post(text)
                            video_url = None
                        else:
                            image_url = _media_url_for_text_post(category)
                            video_url = None
                        feed_items.append({
                            "id": str(row.name),
                            "headline": text,
                            "text": text,
                            "category": category,
                            "modal": modal,
                            "source": str(row.get("source", "smart-feed")),
                            "image_url": image_url,
                            "video_url": video_url,
                            "toxicity_score": 0.0,
                            "score": 0.5,
                            "score_detail": {},
                            "explanation": {"reasons": ["Catégorie " + category], "summary": "Contenu de la catégorie " + category, "detail": {}, "score": 0.5},
                        })
            except Exception:
                pass

    return {
        "user_id": user_id,
        "version": version,
        "ranking_method": ranking_method,
        "mode": effective_mode,
        "allowed_categories": allowed_categories,
        "count": len(feed_items),
        "feed": feed_items,
    }