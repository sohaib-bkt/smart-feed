import numpy as np
from app.models.embeddings import embed_text


ACTION_WEIGHTS = {
    "like": 1.0,
    "watch_full": 0.8,
    "watch_50": 0.4,
    "comment": 0.6,
    "skip": -0.3
}


def compute_user_embedding(interactions: list[dict]) -> list[float]:
    """
    Compute user embedding from interaction history.
    
    Args:
        interactions: List of dicts with keys "embedding" and "action"
        Example: [{"embedding": [...], "action": "like"}, ...]
    
    Returns:
        Normalized 384-dimensional embedding vector as list[float]
    """
    if not interactions:
        return [0.0] * 384
    
    weighted_vecs, total_w = [], 0
    for item in interactions[-100:]:   # Last 100 interactions
        w = ACTION_WEIGHTS.get(item["action"], 0)
        if w != 0:
            weighted_vecs.append(np.array(item["embedding"]) * w)
            total_w += abs(w)
    
    if not weighted_vecs or total_w == 0:
        return [0.0] * 384
    
    user_emb = np.sum(weighted_vecs, axis=0) / total_w
    # Normalize
    norm = np.linalg.norm(user_emb)
    if norm > 0:
        user_emb = user_emb / norm
    return user_emb.tolist()
