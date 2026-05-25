# app/services/multimodal_index.py
"""
Service d'index multimodal FAISS v3

Gère :
- Chargement de l'index FAISS v3 (texte + images + vidéos)
- Recherche cross-modale (texte → image/vidéo)
- Filtrage par modalité
- Projection des embeddings texte (384d → 512d)

Utilisation:
    from app.services.multimodal_index import get_v3_index, search_multimodal

    index, metadata = get_v3_index()
    results = search_multimodal(query_embedding, k=20, modal_filter='image')
"""

import logging
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal
from functools import lru_cache

logger = logging.getLogger(__name__)

# Types de modalités supportées
ModalType = Literal['text', 'image', 'video', 'all']

# Chemins des fichiers
INDEX_PATH = Path("data/processed/faiss_index_v3.bin")
METADATA_PATH = Path("data/processed/combined_meta_v3.parquet")
PROJ_PATH = Path("data/processed/proj_384_512.npy")
PROJ_VIDEO_PATH = Path("data/processed/proj_video_500_512.npy")


# ============================================================================
# Chargement des ressources (caché pour performance)
# ============================================================================

@lru_cache(maxsize=1)
def _load_index() -> Tuple[faiss.Index, pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Charge l'index FAISS v3 et ses métadonnées.
    Résultat mis en cache pour les appels suivants.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS v3 index not found at {INDEX_PATH}. "
                                f"Run scripts/phase3/build_multimodal_index.py first.")
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
    
    # Charger l'index
    index = faiss.read_index(str(INDEX_PATH))
    logger.info(f"✅ FAISS v3 index loaded: {index.ntotal} vectors, dimension {index.d}")
    
    # Charger les métadonnées
    metadata = pd.read_parquet(METADATA_PATH)
    logger.info(f"✅ Metadata loaded: {len(metadata)} rows, columns: {metadata.columns.tolist()}")
    
    # Charger les matrices de projection (si disponibles)
    proj_text = None
    if PROJ_PATH.exists():
        proj_text = np.load(PROJ_PATH).astype('float32')
        logger.info(f"✅ Text projection matrix loaded: {proj_text.shape}")
    
    proj_video = None
    if PROJ_VIDEO_PATH.exists():
        proj_video = np.load(PROJ_VIDEO_PATH).astype('float32')
        logger.info(f"✅ Video projection matrix loaded: {proj_video.shape}")
    
    return index, metadata, proj_text, proj_video


def get_v3_index() -> Tuple[faiss.Index, pd.DataFrame]:
    """Retourne l'index FAISS v3 et ses métadonnées."""
    index, metadata, _, _ = _load_index()
    return index, metadata


def get_projection_matrices() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Retourne les matrices de projection (texte, vidéo)."""
    _, _, proj_text, proj_video = _load_index()
    return proj_text, proj_video


# ============================================================================
# Projection d'embeddings
# ============================================================================

def project_text_embedding(embedding_384: np.ndarray) -> np.ndarray:
    """
    Projette un embedding texte SBERT (384d) vers l'espace multimodal (512d).
    
    Args:
        embedding_384: Embedding de dimension 384 (de SBERT)
    
    Returns:
        Embedding projeté de dimension 512, normalisé L2
    """
    proj_text, _ = get_projection_matrices()
    if proj_text is None:
        raise RuntimeError("Text projection matrix not available. "
                          f"Run scripts/phase3/build_multimodal_index.py first.")
    
    if isinstance(embedding_384, list):
        embedding_384 = np.array(embedding_384)
    
    if embedding_384.ndim == 1:
        embedding_384 = embedding_384.reshape(1, -1)
    
    # Projection
    emb_512 = (embedding_384.astype('float32') @ proj_text).astype('float32')
    
    # Normalisation L2
    norms = np.linalg.norm(emb_512, axis=1, keepdims=True).clip(min=1e-8)
    emb_512 = emb_512 / norms
    
    return emb_512


def project_video_embedding(embedding_500: np.ndarray) -> np.ndarray:
    """
    Projette un embedding vidéo VideoMAE (500d) vers l'espace multimodal (512d).
    
    Args:
        embedding_500: Embedding de dimension 500 (de VideoMAE)
    
    Returns:
        Embedding projeté de dimension 512, normalisé L2
    """
    _, proj_video = get_projection_matrices()
    if proj_video is None:
        raise RuntimeError("Video projection matrix not available.")
    
    if isinstance(embedding_500, list):
        embedding_500 = np.array(embedding_500)
    
    if embedding_500.ndim == 1:
        embedding_500 = embedding_500.reshape(1, -1)
    
    # Projection (500 -> 512)
    emb_512 = (embedding_500.astype('float32') @ proj_video).astype('float32')
    
    # Normalisation L2
    norms = np.linalg.norm(emb_512, axis=1, keepdims=True).clip(min=1e-8)
    emb_512 = emb_512 / norms
    
    return emb_512


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalise un embedding L2 (pour ceux déjà dans l'espace 512d)."""
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    norms = np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-8)
    return embedding / norms


# ============================================================================
# Recherche multimodale
# ============================================================================

def search_multimodal(
    query_embedding: np.ndarray,
    k: int = 20,
    modal_filter: ModalType = 'all',
    source_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    exclude_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Recherche dans l'index multimodal.
    
    Args:
        query_embedding: Embedding de la requête (déjà dans l'espace 512d)
        k: Nombre de résultats à retourner
        modal_filter: Filtrer par modalité ('text', 'image', 'video', 'all')
        source_filter: Filtrer par source ('huffpost', 'reddit', 'coco', 'video')
        category_filter: Filtrer par catégorie
        exclude_ids: IDs à exclure des résultats
    
    Returns:
        Liste de résultats avec scores et métadonnées
    """
    index, metadata = get_v3_index()
    
    # Normaliser l'embedding
    query = normalize_embedding(query_embedding)
    
    # Recherche (chercher plus pour permettre filtrage)
    search_k = min(k * 3, index.ntotal)
    scores, indices = index.search(query, search_k)
    
    results = []
    seen_ids = set()
    excluded = set(exclude_ids or [])
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        row = metadata.iloc[idx]
        item_id = row.get('id', idx)
        
        # Éviter les doublons
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        
        # Exclure les IDs spécifiques
        if item_id in excluded:
            continue
        
        # Filtre par modalité
        modal = row.get('modal', 'text')
        if modal_filter != 'all' and modal != modal_filter:
            continue
        
        # Filtre par source
        if source_filter and row.get('source') != source_filter:
            continue
        
        # Filtre par catégorie
        if category_filter and row.get('category') != category_filter:
            continue
        
        results.append({
            'id': str(item_id),
            'score': float(score),
            'text': str(row.get('text', ''))[:500],
            'category': str(row.get('category', 'unknown')),
            'modal': modal,
            'source': str(row.get('source', 'unknown')),
            'metadata': row.to_dict()
        })
        
        if len(results) >= k:
            break
    
    return results


def search_by_text(
    text_embedding_384: np.ndarray,
    k: int = 20,
    modal_filter: ModalType = 'all',
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Recherche à partir d'un embedding texte (384d).
    Projette automatiquement vers l'espace multimodal.
    
    Args:
        text_embedding_384: Embedding SBERT (384d)
        k: Nombre de résultats
        modal_filter: Filtrer par modalité
        **kwargs: Arguments supplémentaires pour search_multimodal
    
    Returns:
        Liste des résultats
    """
    query_512 = project_text_embedding(text_embedding_384)
    return search_multimodal(query_512, k=k, modal_filter=modal_filter, **kwargs)


def search_by_image(
    image_embedding_512: np.ndarray,
    k: int = 20,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Recherche à partir d'un embedding image (CLIP, déjà en 512d).
    
    Args:
        image_embedding_512: Embedding CLIP (512d)
        k: Nombre de résultats
        **kwargs: Arguments supplémentaires pour search_multimodal
    
    Returns:
        Liste des résultats
    """
    return search_multimodal(image_embedding_512, k=k, **kwargs)


def search_by_video(
    video_embedding_500: np.ndarray,
    k: int = 20,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Recherche à partir d'un embedding vidéo (VideoMAE, 500d).
    Projette automatiquement vers l'espace multimodal.
    
    Args:
        video_embedding_500: Embedding VideoMAE (500d)
        k: Nombre de résultats
        **kwargs: Arguments supplémentaires pour search_multimodal
    
    Returns:
        Liste des résultats
    """
    query_512 = project_video_embedding(video_embedding_500)
    return search_multimodal(query_512, k=k, **kwargs)


# ============================================================================
# Statistiques et informations sur l'index
# ============================================================================

def get_index_stats() -> Dict[str, Any]:
    """Retourne des statistiques sur l'index multimodal."""
    index, metadata = get_v3_index()
    
    stats = {
        'total_vectors': index.ntotal,
        'dimension': index.d,
        'nprobe': getattr(index, 'nprobe', 1),
    }
    
    # Statistiques par modalité
    if 'modal' in metadata.columns:
        modal_counts = metadata['modal'].value_counts().to_dict()
        stats['by_modal'] = modal_counts
    
    # Statistiques par source
    if 'source' in metadata.columns:
        source_counts = metadata['source'].value_counts().to_dict()
        stats['by_source'] = source_counts
    
    # Catégories uniques
    if 'category' in metadata.columns:
        stats['unique_categories'] = metadata['category'].nunique()
    
    return stats


def get_random_examples(modal: Optional[ModalType] = None, n: int = 5) -> pd.DataFrame:
    """Retourne des exemples aléatoires de l'index."""
    _, metadata = get_v3_index()
    
    if modal and modal != 'all':
        filtered = metadata[metadata.get('modal', 'text') == modal]
    else:
        filtered = metadata
    
    return filtered.sample(min(n, len(filtered)))


# ============================================================================
# Fonction utilitaire pour les tests
# ============================================================================

def test_index_health() -> Dict[str, bool]:
    """Vérifie que tous les composants de l'index fonctionnent."""
    results = {}
    
    # Vérifier les fichiers
    results['index_exists'] = INDEX_PATH.exists()
    results['metadata_exists'] = METADATA_PATH.exists()
    
    if not results['index_exists']:
        return results
    
    try:
        index, metadata, proj_text, proj_video = _load_index()
        results['index_loaded'] = True
        results['total_vectors'] = index.ntotal
        results['has_metadata'] = len(metadata) > 0
        results['has_proj_text'] = proj_text is not None
        results['has_proj_video'] = proj_video is not None
        
        # Vérifier les modalités
        if 'modal' in metadata.columns:
            modal_counts = metadata['modal'].value_counts()
            results['has_text'] = modal_counts.get('text', 0) > 0
            results['has_images'] = modal_counts.get('image', 0) > 0
            results['has_videos'] = modal_counts.get('video', 0) > 0
        
        # Test de recherche simple
        test_query = np.random.randn(1, index.d).astype('float32')
        test_query = test_query / np.linalg.norm(test_query)
        scores, indices = index.search(test_query, 5)
        results['search_works'] = len(scores[0]) == 5
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


# ============================================================================
# Initialisation au chargement du module
# ============================================================================

def init():
    """Initialise le module (préchauffe le cache)."""
    try:
        get_v3_index()
        logger.info("✅ Multimodal index service initialized")
    except Exception as e:
        logger.warning(f"⚠️ Multimodal index not available: {e}")


# Auto-initialisation
init()