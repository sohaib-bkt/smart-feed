# app/services/hybrid_recommender.py (fixed version)

import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from app.models.collaborative import get_cf_scores
from app.services.scoring import compute_score

_index = None
_df = None
_position_to_id = None
_id_to_position = None

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


def get_hybrid_feed(
    user_id: str,
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20
) -> list[dict]:
    """Feed hybride = content-based (FAISS) + collaborative (Implicit)."""
    _load()
    
    ALPHA = 0.6
    BETA = 0.4
    
    # Convert to numpy array
    if isinstance(user_embedding, list):
        query = np.array([user_embedding], dtype="float32")
    else:
        query = np.array([user_embedding], dtype="float32")
    
    # FAISS search
    sims, positions = _index.search(query, n_candidates)
    
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
        
        # FIX: Safely extract text with proper string conversion
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
    
    results.sort(key=lambda x: x["score"], reverse=True)
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