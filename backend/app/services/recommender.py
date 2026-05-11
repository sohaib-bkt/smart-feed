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