# app/services/hybrid_recommender.py (version avec diversification)

import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from app.models.collaborative import get_cf_scores
from app.services.recommender import get_feed_v2
from app.services.scoring import compute_score

_index = None
_df = None
_position_to_id = None
_id_to_position = None

# Paramètres de diversification
MAX_PER_CATEGORY = 3  # Maximum d'articles par catégorie
MIN_CATEGORIES = 3    # Nombre minimum de catégories à essayer d'inclure

def _load():
    global _index, _df, _position_to_id, _id_to_position
    
    if _index is None:
        # Load FAISS index
        _index = faiss.read_index("data/processed/faiss_index_v2.bin")
        _index.nprobe = 10
        
        # Load data
        try:
            df_hp = pd.read_parquet("data/processed/huffpost_with_meta.parquet")
            df_rd = pd.read_parquet("data/raw/reddit.parquet")
            _df = pd.concat([df_hp, df_rd], ignore_index=True)
        except:
            _df = pd.read_parquet("data/processed/huffpost_with_meta.parquet")
        
        # Convert date columns to datetime
        if 'date' in _df.columns:
            _df['date'] = pd.to_datetime(_df['date'], errors='coerce')
        elif 'created_at' in _df.columns:
            _df['created_at'] = pd.to_datetime(_df['created_at'], errors='coerce')
        
        # Convert text columns to string to avoid float errors
        if 'headline' in _df.columns:
            _df['headline'] = _df['headline'].fillna('').astype(str)
        if 'text' in _df.columns:
            _df['text'] = _df['text'].fillna('').astype(str)
        
        # Create mapping
        if 'id' in _df.columns:
            article_ids = _df['id'].tolist()
        elif 'article_id' in _df.columns:
            article_ids = _df['article_id'].tolist()
        else:
            article_ids = _df.index.tolist()
        
        _position_to_id = {i: str(aid) for i, aid in enumerate(article_ids)}
        _id_to_position = {str(aid): i for i, aid in enumerate(article_ids)}
        
        print(f"Index hybride chargé : {_index.ntotal} articles")


def _safe_get_text(post: dict, key: str, default: str = "") -> str:
    """Safely extract text from post dict, handling non-string values."""
    value = post.get(key, default)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return default


def _diversify_results(results: list[dict], max_per_category: int = MAX_PER_CATEGORY) -> list[dict]:
    """
    Diversifie les résultats en limitant le nombre d'articles par catégorie.
    
    Args:
        results: Liste des articles triés par score
        max_per_category: Nombre maximum d'articles par catégorie
    
    Returns:
        Liste diversifiée (même ordre de score, mais limitée par catégorie)
    """
    if not results:
        return []
    
    diversified = []
    category_count = defaultdict(int)
    
    for item in results:
        category = item.get('category', 'Unknown')
        
        # Ajouter l'article si la catégorie n'a pas atteint sa limite
        if category_count[category] < max_per_category:
            diversified.append(item)
            category_count[category] += 1
    
    return diversified


def _collab_artifacts_available() -> bool:
    backend_root = Path(__file__).resolve().parents[2]
    model_path = backend_root / "app/models/implicit_model.pkl"
    matrix_path = backend_root / "data/processed/interaction_matrix.npz"
    return model_path.exists() and matrix_path.exists()


def get_hybrid_feed(
    user_id: str,
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20,
    diversify: bool = True,
    max_per_category: int = MAX_PER_CATEGORY
) -> list[dict]:
    """Feed hybride = content-based (FAISS) + collaborative (Implicit)."""
    if not _collab_artifacts_available():
        return get_feed_v2(
            user_embedding=user_embedding,
            user_prefs=user_prefs,
            n_candidates=n_candidates,
            n_results=n_results,
        )
    _load()
    
    ALPHA = 0.6
    BETA = 0.4
    
    # Convert to numpy array
    if isinstance(user_embedding, list):
        query = np.array([user_embedding], dtype="float32")
    else:
        query = np.array([user_embedding], dtype="float32")
    
    # FAISS search (prendre plus de candidats pour permettre la diversification)
    search_n = n_candidates * 2 if diversify else n_candidates
    sims, positions = _index.search(query, search_n)
    
    # Get candidate article IDs
    candidate_positions = [int(p) for p in positions[0] if p >= 0]
    candidate_article_ids = []
    for pos in candidate_positions:
        if pos in _position_to_id:
            candidate_article_ids.append(int(_position_to_id[pos]))
    
    if not candidate_article_ids:
        return []
    
    # Get collaborative scores
    cf_scores = get_cf_scores(user_id, candidate_article_ids)
    
    results = []
    for sim, pos in zip(sims[0], positions[0]):
        if pos < 0 or pos >= len(_df):
            continue
        
        article_id = _position_to_id.get(int(pos))
        if article_id is None:
            continue
        
        post = _df.iloc[int(pos)].to_dict()
        cf_score = cf_scores.get(int(article_id), 0.5)
        
        # Handle date properly
        post_date = post.get("date") or post.get("created_at")
        if post_date and hasattr(post_date, 'strftime'):
            pass
        elif post_date and isinstance(post_date, str):
            try:
                post_date = pd.to_datetime(post_date)
            except:
                post_date = None
        
        cb_scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get("toxicity_score", 0.0),
            category=post.get("category", ""),
            user_interests=user_prefs.get("interests", []),
            created_at=post_date,
            user_prefs=user_prefs
        )
        
        if cb_scored.get("filtered", False):
            continue
        
        cb_score = cb_scored["score"]
        hybrid_score = (ALPHA * cb_score) + (BETA * cf_score)
        
        # Safely extract text
        headline = _safe_get_text(post, "headline")
        if not headline:
            headline = _safe_get_text(post, "text")
        if not headline:
            headline = f"Article {article_id}"
        
        results.append({
            "id": str(article_id),
            "headline": headline[:200],
            "category": str(post.get("category", "General"))[:50],
            "score": round(hybrid_score, 6),
            "score_detail": {
                "content_based": round(cb_score, 4),
                "collaborative": round(cf_score, 4),
                "cosine_sim": round(float(sim), 4)
            },
            "explanation": _build_explanation(float(sim), cf_score, post, user_prefs)
        })
    
    # Trier par score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Appliquer la diversification si demandée
    if diversify:
        results = _diversify_results(results, max_per_category)
    
    return results[:n_results]


def _build_explanation(sim: float, cf_score: float, post: dict, prefs: dict) -> str:
    reasons = []
    
    if sim > 0.6:
        reasons.append(f"très similaire à vos lectures ({sim:.0%})")
    elif sim > 0.4:
        reasons.append("similar à vos intérêts")
    
    if cf_score > 0.7:
        reasons.append("très apprécié par des lecteurs similaires")
    elif cf_score > 0.6:
        reasons.append("apprécié par des lecteurs similaires")
    
    category = post.get("category", "")
    if category and category in prefs.get("interests", []):
        reasons.append(f"catégorie {category} que vous aimez")
    
    if reasons:
        return "Recommandé car : " + ", ".join(reasons)
    return "Recommandé pour vous"