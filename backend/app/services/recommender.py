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
    include_images: bool = True,
    include_videos: bool = True
) -> list[dict]:
    """
    Feed multimodal v3 avec support images et vidéos.
    """
    _load_resources_v3()
    
    # Projeter l'embedding utilisateur
    if embedding_dim == 384 and _proj_384_512 is not None:
        query_emb = np.array([user_embedding], dtype='float32')
        query_emb = query_emb @ _proj_384_512
    else:
        query_emb = np.array([user_embedding], dtype='float32')
    
    # Normaliser
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm
    
    # Recherche FAISS (prendre plus de candidats pour diversité)
    search_k = n_candidates * 2
    scores, indices = _index_v3.search(query_emb, search_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_posts_df_v3):
            continue
        
        post = _posts_df_v3.iloc[idx].to_dict()
        modal = post.get('modal', 'text')
        
        # Filtrer par modalité
        if modal == 'image' and not include_images:
            continue
        if modal == 'video' and not include_videos:
            continue
        
        # Récupérer le texte
        text = post.get('text', '')
        if not text or text == 'None' or str(text) == 'nan':
            text = post.get('caption', '')
        if not text or text == 'None':
            text = f"{modal} content"
        
        # Convertir en string proprement
        text = str(text) if text else ""
        
        results.append({
            'id': str(post.get('id', idx)),
            'text': text[:500],
            'category': str(post.get('category', 'general')),
            'modal': modal,
            'source': str(post.get('source', 'unknown')),
            'score': float(score),
            'score_detail': {
                'similarity': float(score),
                'modal': modal
            },
            'explanation': f"Recommandé car : {modal} similaire à vos intérêts"
        })
    
    # Diversifier les modalités si demandé
    if include_images and include_videos and len(results) > n_results:
        text_results = [r for r in results if r['modal'] == 'text']
        image_results = [r for r in results if r['modal'] == 'image']
        video_results = [r for r in results if r['modal'] == 'video']
        
        # Construire un feed diversifié
        diversified = []
        diversified.extend(text_results[:n_results//2])
        diversified.extend(image_results[:n_results//3])
        diversified.extend(video_results[:n_results//6])
        
        # Compléter avec les meilleurs restants
        remaining = [r for r in results if r not in diversified]
        diversified.extend(remaining[:n_results - len(diversified)])
        
        diversified.sort(key=lambda x: x['score'], reverse=True)
        results = diversified
    
    return results[:n_results]