"""
ranker.py : Formule de scoring pondérée (Phase 2)

Score final = w_sim   * cosine_similarity(user_emb, post_emb)
            + w_clean * (1 - toxicity_score)
            + w_pop   * popularity_score(likes, views)
            + w_fresh * freshness_score(created_at)

Les poids sont configurables via RANKER_WEIGHTS.
Les embeddings sont supposés déjà normalisés (L2=1),
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Poids de la formule — doivent sommer à 1.0
RANKER_WEIGHTS: dict[str, float] = {
    "similarity": 0.45,   
    "clean":      0.25,   # pénalité toxicité  
    "popularity": 0.15,   # engagement social  (likes + views)
    "freshness":  0.15,   # fraicheur du post  
}

# (après N jours → score divisé par 2)
FRESHNESS_HALF_LIFE_DAYS: float = 7.0

# Popularité de référence pour normalisation log : log1p(POP_SCALE)
POP_SCALE: float = 10_000.0


# Fonctions de sous-score (chacune retourne un float dans [0, 1])

def _similarity_score(user_emb: list[float], post_emb: list[float]) -> float:
    """
    Cosine similarity entre l'embedding utilisateur et celui du post.
    Les embeddings étant normalisés (all-MiniLM-L6-v2 + normalize_embeddings=True),
    le cosine sim = dot product, ramené dans [0, 1] via (x + 1) / 2.

    Retourne 0.5 si l'un des embeddings est nul (utilisateur sans historique).
    """
    u = np.array(user_emb, dtype=np.float32)
    p = np.array(post_emb, dtype=np.float32)

    norm_u = np.linalg.norm(u)
    norm_p = np.linalg.norm(p)

    if norm_u == 0.0 or norm_p == 0.0:
        return 0.5  # neutre — pas d'info suffisante

    cos_sim = float(np.dot(u, p) / (norm_u * norm_p))
    cos_sim = max(-1.0, min(1.0, cos_sim))  # clip flottants
    return (cos_sim + 1.0) / 2.0            # [−1,1] → [0,1]


def _clean_score(toxicity_score: float) -> float:
    """
    Score de propreté = 1 − toxicity_score.
    Un post toxique est pénalisé ; un post sain est favorisé.
    """
    return 1.0 - max(0.0, min(1.0, toxicity_score))


def _popularity_score(likes: int, views: int) -> float:
    """
    Score de popularité normalisé en log.
    Formule : log1p(likes + views * 0.1) / log1p(POP_SCALE)
    Le facteur 0.1 évite que les vues dominent complètement les likes.
    Plafonné à 1.0.
    """
    raw = math.log1p(likes + views * 0.1)
    return min(1.0, raw / math.log1p(POP_SCALE))


def _freshness_score(created_at: Optional[datetime]) -> float:
    """
    Décroissance exponentielle basée sur l'âge du post.
    score = exp(−λ × jours_depuis_publication)
    où λ = ln(2) / FRESHNESS_HALF_LIFE_DAYS.

    Retourne 1.0 si created_at est absent (pas de pénalité).
    """
    if created_at is None:
        return 1.0

    now = datetime.now(timezone.utc)
    # S'assurer que created_at est timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - created_at).total_seconds() / 86_400)
    lam = math.log(2) / FRESHNESS_HALF_LIFE_DAYS
    return math.exp(-lam * age_days)


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def score_candidate(
    candidate: dict,
    user_embedding: list[float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Calcule le score final d'un candidat pour un utilisateur donné.

    Args:
        candidate: dict avec les clés :
            - embedding       : list[float] (384-dim, normalisé)
            - toxicity_score  : float ∈ [0, 1]
            - likes           : int
            - views           : int
            - created_at      : datetime | None
        user_embedding: vecteur utilisateur (384-dim, normalisé)
        weights: override de RANKER_WEIGHTS si fourni

    Returns:
        Score final ∈ [0, 1]
    """
    w = weights or RANKER_WEIGHTS

    post_emb      = candidate.get("embedding") or []
    toxicity      = float(candidate.get("toxicity_score", 0.0))
    likes         = int(candidate.get("likes", 0))
    views         = int(candidate.get("views", 0))
    created_at    = candidate.get("created_at")

    sim   = _similarity_score(user_embedding, post_emb)
    clean = _clean_score(toxicity)
    pop   = _popularity_score(likes, views)
    fresh = _freshness_score(created_at)

    score = (
        w["similarity"] * sim
        + w["clean"]      * clean
        + w["popularity"] * pop
        + w["freshness"]  * fresh
    )

    logger.debug(
        "id=%s sim=%.3f clean=%.3f pop=%.3f fresh=%.3f → %.4f",
        candidate.get("id", "?"), sim, clean, pop, fresh, score,
    )

    return round(score, 6)


def rank_candidates(
    candidates: list[dict],
    user_embedding: list[float] | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Trie une liste de candidats par score décroissant.

    Chaque candidat reçoit un champ `rank_score` (float).
    Si `user_embedding` est None, l'embedding nul est utilisé
    (similitude neutre = 0.5 pour tous les posts).

    Args:
        candidates     : liste de dicts (voir score_candidate)
        user_embedding : vecteur utilisateur 384-dim
        weights        : poids custom (optionnel)

    Returns:
        Liste triée par rank_score décroissant, avec le champ ajouté.
    """
    if not candidates:
        return []

    emb = user_embedding or [0.0] * 384

    scored = []
    for c in candidates:
        c = dict(c)  # copie pour ne pas muter l'original
        c["rank_score"] = score_candidate(c, emb, weights)
        scored.append(c)

    scored.sort(key=lambda x: x["rank_score"], reverse=True)
    return scored