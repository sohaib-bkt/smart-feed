# app/services/cross_modal_search.py
"""
Service de recherche cross-modale (texte → image/vidéo)
Utilise CLIP pour aligner texte et images/vidéos.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models.embeddings import embed_text
from app.services.multimodal_index import get_v3_index, search_multimodal

# Matrice de projection (chargée une fois)
_proj_384_512 = None
_clip_model_loaded = False

def _get_projection():
    global _proj_384_512
    if _proj_384_512 is None:
        proj_path = Path("data/processed/proj_384_512.npy")
        if proj_path.exists():
            _proj_384_512 = np.load(proj_path).astype('float32')
    return _proj_384_512


def _get_clip_embedding(text: str) -> np.ndarray:
    """Extrait un embedding texte via CLIP."""
    global _clip_model_loaded
    
    try:
        from app.models.multimodal.clip_model import embed_post_multimodal
        
        # Créer une image factice pour extraire l'embedding texte
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
        embedding = embed_post_multimodal(text, dummy_image, alpha=1.0)
        return np.array(embedding)
    except Exception as e:
        print(f"⚠️ CLIP not available: {e}, falling back to SBERT")
        _clip_model_loaded = False
        return None


def search_by_text_cross_modal(
    query_text: str,
    k: int = 20,
    modal_filter: str = 'all',  # 'all', 'image', 'video'
    use_clip: bool = True
) -> List[Dict[str, Any]]:
    """
    Recherche cross-modale : texte → images/vidéos.
    
    Args:
        query_text: Requête texte
        k: Nombre de résultats
        modal_filter: Filtrer par modalité ('image', 'video', 'all')
        use_clip: Utiliser CLIP (True) ou SBERT projeté (False)
    
    Returns:
        Liste de résultats avec métadonnées
    """
    # Obtenir l'embedding de la requête
    if use_clip:
        query_emb = _get_clip_embedding(query_text)
        if query_emb is None:
            use_clip = False
    
    if not use_clip:
        # Fallback: SBERT + projection
        query_emb = embed_text(query_text)
        proj = _get_projection()
        if proj is not None:
            query_emb = query_emb @ proj
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    # Recherche dans l'index multimodal
    results = search_multimodal(query_emb, k=k, modal_filter=modal_filter)
    
    return results


def search_similar_images(
    image_embedding: np.ndarray,
    k: int = 20
) -> List[Dict[str, Any]]:
    """
    Recherche d'images similaires à partir d'un embedding CLIP.
    
    Args:
        image_embedding: Embedding CLIP de l'image (512d)
        k: Nombre de résultats
    
    Returns:
        Liste d'articles similaires
    """
    # Normaliser l'embedding
    if isinstance(image_embedding, list):
        image_embedding = np.array(image_embedding)
    
    query_emb = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    
    # Recherche dans l'index multimodal
    results = search_multimodal(query_emb, k=k, modal_filter='image')
    
    return results