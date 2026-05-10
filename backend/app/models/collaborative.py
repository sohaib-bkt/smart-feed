import numpy as np
import pickle
import json
from functools import lru_cache
from scipy.sparse import load_npz

@lru_cache(maxsize=1)
def _load_model():
    """Charge le modèle Implicit et les mappings"""
    with open('app/models/implicit_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('data/processed/user2idx.json', 'r') as f:
        user2idx = json.load(f)
    with open('data/processed/art2idx.json', 'r') as f:
        art2idx = json.load(f)
    
    matrix = load_npz('data/processed/interaction_matrix.npz')
    
    return model, user2idx, art2idx, matrix


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Normalise les scores bruts pour les étaler entre 0 et 1.
    Utilise min-max scaling au lieu de sigmoid pour de meilleurs résultats.
    """
    if len(raw_scores) == 0:
        return raw_scores
    
    # Option 1: Min-max scaling (recommended)
    min_score = np.min(raw_scores)
    max_score = np.max(raw_scores)
    
    if max_score > min_score:
        normalized = (raw_scores - min_score) / (max_score - min_score)
    else:
        # Fallback to sigmoid if all scores are equal
        normalized = 1.0 / (1.0 + np.exp(-raw_scores))
    
    return normalized


def get_cf_scores(user_id: str, article_indices: list[int]) -> dict[int, float]:
    """
    Calcule les scores collaboratifs avec le modèle Implicit
    pour un utilisateur sur une liste d'indices d'articles.
    """
    model, user2idx, art2idx, matrix = _load_model()
    
    if str(user_id) not in user2idx:
        return {idx: 0.5 for idx in article_indices}
    
    u_idx = user2idx[str(user_id)]
    
    if u_idx >= model.user_factors.shape[0]:
        return {idx: 0.5 for idx in article_indices}
    
    valid_articles = []
    valid_indices = []
    
    for art_idx in article_indices:
        art_key = str(art_idx)
        if art_key in art2idx:
            a_idx = art2idx[art_key]
            if a_idx < model.item_factors.shape[0]:
                valid_articles.append(art_idx)
                valid_indices.append(a_idx)
    
    if not valid_indices:
        return {idx: 0.5 for idx in article_indices}
    
    try:
        # Calculer les scores via dot product
        user_factors = model.user_factors[u_idx]
        item_factors = model.item_factors[valid_indices]
        raw_scores = np.dot(item_factors, user_factors)
        
        # Appliquer un scaling plus agressif
        # Multiplier par 10 pour amplifier les différences
        scaled_scores = raw_scores * 10
        
        # Normaliser avec sigmoid après scaling
        normalized_scores = 1.0 / (1.0 + np.exp(-scaled_scores))
        
        scores = {}
        for art_idx, norm_score in zip(valid_articles, normalized_scores):
            scores[art_idx] = float(norm_score)
        
        return scores
    
    except Exception as e:
        print(f"Erreur lors du calcul des scores: {e}")
        return {idx: 0.5 for idx in article_indices}


def get_top_cf_articles(user_id: str, n: int = 50) -> list[int]:
    """Retourne les N articles avec le score CF le plus élevé"""
    model, user2idx, art2idx, matrix = _load_model()
    
    if str(user_id) not in user2idx:
        return []
    
    u_idx = user2idx[str(user_id)]
    
    if u_idx >= model.user_factors.shape[0]:
        return []
    
    try:
        user_interactions = matrix[u_idx]
        
        recommendations = model.recommend(
            u_idx,
            user_interactions,
            N=n,
            filter_already_liked_items=True
        )
        
        idx2art = {v: int(k) for k, v in art2idx.items()}
        
        top_articles = []
        for item_idx in recommendations[0]:
            if item_idx in idx2art:
                top_articles.append(idx2art[item_idx])
        
        return top_articles
    
    except Exception as e:
        print(f"Erreur: {e}")
        return []


def get_user_recommendations_with_scores(user_id: str, n: int = 50) -> list[tuple[int, float]]:
    """Retourne les N articles recommandés avec leurs scores normalisés"""
    model, user2idx, art2idx, matrix = _load_model()
    
    if str(user_id) not in user2idx:
        return []
    
    u_idx = user2idx[str(user_id)]
    
    if u_idx >= model.user_factors.shape[0]:
        return []
    
    try:
        user_interactions = matrix[u_idx]
        
        recommendations = model.recommend(
            u_idx,
            user_interactions,
            N=n,
            filter_already_liked_items=True
        )
        
        idx2art = {v: int(k) for k, v in art2idx.items()}
        
        results = []
        for item_idx, raw_score in zip(recommendations[0], recommendations[1]):
            if item_idx in idx2art:
                # Amplifier le score pour mieux distinguer
                amplified_score = raw_score * 10
                # Appliquer sigmoid
                normalized_score = float(1.0 / (1.0 + np.exp(-amplified_score)))
                results.append((idx2art[item_idx], normalized_score))
        
        # Normaliser les scores entre 0 et 1 pour meilleure lisibilité
        if results:
            scores = [s for _, s in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score > min_score:
                results = [(aid, (s - min_score) / (max_score - min_score)) 
                          for aid, s in results]
        
        return results
    
    except Exception as e:
        print(f"Erreur: {e}")
        return []


def get_user_recommendations_raw(user_id: str, n: int = 50) -> list[tuple[int, float]]:
    """
    Retourne les scores BRUTS du modèle (sans normalisation).
    Utile pour le debugging.
    """
    model, user2idx, art2idx, matrix = _load_model()
    
    if str(user_id) not in user2idx:
        return []
    
    u_idx = user2idx[str(user_id)]
    
    if u_idx >= model.user_factors.shape[0]:
        return []
    
    try:
        user_interactions = matrix[u_idx]
        
        recommendations = model.recommend(
            u_idx,
            user_interactions,
            N=n,
            filter_already_liked_items=True
        )
        
        idx2art = {v: int(k) for k, v in art2idx.items()}
        
        results = []
        for item_idx, raw_score in zip(recommendations[0], recommendations[1]):
            if item_idx in idx2art:
                results.append((idx2art[item_idx], float(raw_score)))
        
        return results
    
    except Exception as e:
        print(f"Erreur: {e}")
        return []