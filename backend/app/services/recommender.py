import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from app.services.scoring import compute_score
from app.config import settings

# Charger index et métadonnées au démarrage (singleton)
_index = None
_posts_df = None
_index_v2 = None
_posts_df_v2 = None

def _load_resources():
    """Charge l'index FAISS et les métadonnées des posts."""
    global _index, _posts_df
    if _index is None:
        _index = faiss.read_index('data/processed/faiss_index.bin')
        _posts_df = pd.read_parquet('data/processed/huffpost_with_meta.parquet')
        print(f'FAISS index chargé : {_index.ntotal} vecteurs')


def _load_resources_v2():
    """Charge l'index FAISS v2 et les métadonnées combinées des posts."""
    global _index_v2, _posts_df_v2
    if _index_v2 is None:
        _index_v2 = faiss.read_index('data/processed/faiss_index_v2.bin')
        _index_v2.nprobe = 10
        _posts_df_v2 = pd.read_parquet('data/processed/combined_meta.parquet')
        print(f'FAISS v2 index chargé : {_index_v2.ntotal} vecteurs')

def get_feed(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20
) -> list[dict]:
    """
    Retourne le feed personnalisé pour un utilisateur.
    
    Pipeline :
    1. FAISS recherche les n_candidates posts les plus proches
    2. compute_score() calcule le score final de chacun
    3. Tri par score DESC et retour des n_results meilleurs
    
    Args:
        user_embedding: Vecteur utilisateur (384 dimensions)
        user_prefs: Préférences utilisateur (mode, interests, etc.)
        n_candidates: Nombre de candidats à évaluer
        n_results: Nombre de posts à retourner
    
    Returns:
        Liste de posts scorés, triés par pertinence
    """
    _load_resources()
    
    query = np.array([user_embedding], dtype='float32')
    sims, ids = _index.search(query, n_candidates)
    
    results = []
    user_interests = user_prefs.get('interests', [])
    
    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0 or idx >= len(_posts_df):
            continue
        
        post = _posts_df.iloc[idx].to_dict()
        
        # Convertir la date si nécessaire
        created_at = None
        if 'date' in post and pd.notna(post['date']):
            created_at = pd.Timestamp(post['date']).to_pydatetime()
        
        scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get('toxicity_score', 0.0),
            category=post.get('category', ''),
            user_interests=user_interests,
            created_at=created_at,
            user_prefs=user_prefs
        )
        
        if scored['score'] > 0:
            results.append({
                'id': str(idx),
                'text': post.get('text', '')[:280],
                'category': post.get('category', ''),
                'toxicity_score': post.get('toxicity_score', 0.0),
                'score': scored['score'],
                'score_detail': scored.get('detail', {}),
                'explanation': _build_explanation(scored, post, user_prefs)
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_results]


def get_feed_v2(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20
) -> list[dict]:
    """
    Retourne le feed personnalisé v2 pour un utilisateur.
    Utilise l'index FAISS v2 et les métadonnées combinées.
    """
    _load_resources_v2()

    query = np.array([user_embedding], dtype='float32')
    sims, ids = _index_v2.search(query, n_candidates)

    results = []
    user_interests = user_prefs.get('interests', [])

    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0 or idx >= len(_posts_df_v2):
            continue

        post = _posts_df_v2.iloc[idx].to_dict()

        # Convertir la date si nécessaire
        created_at = None
        if 'date' in post and pd.notna(post['date']):
            created_at = pd.Timestamp(post['date']).to_pydatetime()

        scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get('toxicity_score', 0.0),
            category=post.get('category', ''),
            user_interests=user_interests,
            created_at=created_at,
            user_prefs=user_prefs
        )

        if scored['score'] > 0:
            results.append({
                'id': str(idx),
                'text': post.get('text', '')[:280],
                'category': post.get('category', ''),
                'toxicity_score': post.get('toxicity_score', 0.0),
                'source': post.get('source', ''),
                'score': scored['score'],
                'score_detail': scored.get('detail', {}),
                'explanation': _build_explanation(scored, post, user_prefs)
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_results]

def _build_explanation(scored: dict, post: dict, prefs: dict) -> str:
    """
    Génère une explication lisible pour l'explicabilité du feed.
    """
    d = scored.get('detail', {})
    reasons = []
    
    if d.get('similarity', 0) > 0.5:
        reasons.append('correspond à tes intérêts')
    if post.get('category') in prefs.get('interests', []):
        reasons.append(f'catégorie {post.get("category")} favorite')
    if d.get('recency', 0) > 0.8:
        reasons.append('contenu récent')
    
    return 'Montré car : ' + ', '.join(reasons) if reasons else 'Recommandé'

# -- FAISS V3 globals -----------------------------------------------
_index_v3 = None
_posts_df_v3 = None
_proj_384_512 = None


def _load_resources_v3() -> None:
    """Load V3 index, combined metadata, and the 384->512 projection matrix."""
    global _index_v3, _posts_df_v3, _proj_384_512
    if _index_v3 is None:
        _index_v3 = faiss.read_index("data/processed/faiss_index_v3.bin")
        _index_v3.nprobe = 10
        _posts_df_v3 = pd.read_parquet("data/processed/combined_meta_v3.parquet")
        _proj_384_512 = np.load("data/processed/proj_384_512.npy").astype("float32")
        print(f"FAISS v3 index loaded: {_index_v3.ntotal} vectors at 512-dim")


def _project_384_to_512(embedding_384: list[float]) -> np.ndarray:
    """
    Project a 384-dim MiniLM embedding into 512-dim for V3 index search.
    Used when the user_embedding was computed by MiniLM (standard path).
    """
    _load_resources_v3()
    vec = np.array(embedding_384, dtype="float32").reshape(1, 384)
    vec_512 = vec @ _proj_384_512
    norm = np.linalg.norm(vec_512)
    if norm > 0:
        vec_512 /= norm
    return vec_512.astype("float32")


def get_feed_v3(
    user_embedding: list[float],
    user_prefs: dict,
    n_candidates: int = 200,
    n_results: int = 20,
    embedding_dim: int = 384,
) -> list[dict]:
    """
    Feed using FAISS V3 - unified 512-dim multimodal index.

    Supports:
      - 384-dim user embeddings (MiniLM) -> auto-projected to 512
      - 512-dim user embeddings (CLIP)   -> used directly

    Args:
        user_embedding : user vector from compute_user_embedding()
        user_prefs     : preferences dict (mode, interests, etc.)
        n_candidates   : FAISS candidates to score
        n_results      : posts to return
        embedding_dim  : 384 (MiniLM, default) or 512 (CLIP)

    Returns:
        Ranked list of scored posts with modal + source fields.
    """
    _load_resources_v3()

    if embedding_dim == 384:
        query = _project_384_to_512(user_embedding)
    else:
        query = np.array(user_embedding, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query /= norm

    sims, ids = _index_v3.search(query, n_candidates)

    results = []
    user_interests = user_prefs.get("interests", [])

    for sim, idx in zip(sims[0], ids[0]):
        if idx < 0 or idx >= len(_posts_df_v3):
            continue

        post = _posts_df_v3.iloc[idx].to_dict()

        created_at = None
        if "date" in post and pd.notna(post.get("date")):
            created_at = pd.Timestamp(post["date"]).to_pydatetime()

        scored = compute_score(
            cosine_sim=float(sim),
            toxicity_score=post.get("toxicity_score", 0.0),
            category=post.get("category", ""),
            user_interests=user_interests,
            created_at=created_at,
            user_prefs=user_prefs,
        )

        if scored["score"] > 0:
            results.append(
                {
                    "id": str(idx),
                    "text": post.get("text", "")[:280],
                    "category": post.get("category", ""),
                    "source": post.get("source", ""),
                    "modal": post.get("modal", "text"),
                    "toxicity_score": post.get("toxicity_score", 0.0),
                    "score": scored["score"],
                    "score_detail": scored.get("detail", {}),
                    "explanation": _build_explanation(scored, post, user_prefs),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n_results]