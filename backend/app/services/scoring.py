from datetime import datetime
import math
import numpy as np

# Poids par mode utilisateur
MODE_WEIGHTS = {
    'default':   {'sim': 0.40, 'qual': 0.30, 'pref': 0.20, 'rec': 0.10},
    'focus':     {'sim': 0.60, 'qual': 0.25, 'pref': 0.10, 'rec': 0.05},
    'fun':       {'sim': 0.20, 'qual': 0.20, 'pref': 0.50, 'rec': 0.10},
    'learning':  {'sim': 0.50, 'qual': 0.30, 'pref': 0.15, 'rec': 0.05},
    'fresh':     {'sim': 0.20, 'qual': 0.15, 'pref': 0.15, 'rec': 0.50},  # Added fresh mode
}

def _spread_scores(raw_score: float, mode: str = 'default') -> float:
    """
    Spread compressed scores across 0-1 range.
    Transforms typical 0.5-0.7 range to 0-1 range.
    """
    # Different spreading strategies per mode
    spreading = {
        'default': {'low': 0.5, 'mid_low': 0.6, 'mid_high': 0.7, 'high': 0.8},
        'focus':   {'low': 0.5, 'mid_low': 0.65, 'mid_high': 0.75, 'high': 0.85},
        'fun':     {'low': 0.45, 'mid_low': 0.55, 'mid_high': 0.65, 'high': 0.75},
        'learning':{'low': 0.5, 'mid_low': 0.6, 'mid_high': 0.7, 'high': 0.8},
        'fresh':   {'low': 0.4, 'mid_low': 0.5, 'mid_high': 0.6, 'high': 0.7}
    }
    
    s = spreading.get(mode, spreading['default'])
    
    # Map raw_score to final_score with better distribution
    if raw_score <= s['low']:
        # Very low scores: map to 0-0.2
        final = (raw_score / s['low']) * 0.2
    elif raw_score <= s['mid_low']:
        # Low scores: map to 0.2-0.4
        final = 0.2 + ((raw_score - s['low']) / (s['mid_low'] - s['low'])) * 0.2
    elif raw_score <= s['mid_high']:
        # Medium scores: map to 0.4-0.7
        final = 0.4 + ((raw_score - s['mid_low']) / (s['mid_high'] - s['mid_low'])) * 0.3
    elif raw_score <= s['high']:
        # High scores: map to 0.7-0.95
        final = 0.7 + ((raw_score - s['mid_high']) / (s['high'] - s['mid_high'])) * 0.25
    else:
        # Very high scores: map to 0.95-1.0
        final = 0.95 + ((raw_score - s['high']) / (1.0 - s['high'])) * 0.05
    
    return max(0.01, min(0.99, final))


def _spread_scores_sigmoid(raw_score: float, scale: float = 8.0) -> float:
    """
    Alternative: Use sigmoid to spread scores.
    raw_score: typically 0.5-0.7, scale controls steepness.
    """
    # Center around 0.6 and amplify
    centered = (raw_score - 0.6) * scale
    sigmoid = 1.0 / (1.0 + np.exp(-centered))
    # Rescale from 0.1-0.9 to 0-1
    final = (sigmoid - 0.1) / 0.8
    return max(0.01, min(0.99, final))


def compute_score(
    cosine_sim: float,
    toxicity_score: float,
    category: str,
    user_interests: list[str],
    created_at: datetime | None,
    user_prefs: dict
) -> dict:
    """
    Calcule le score final d'un post pour un utilisateur donné.
    
    Args:
        cosine_sim: Similarité cosinus entre le post et l'utilisateur
        toxicity_score: Score de toxicité (0-1, 0 = sain)
        category: Catégorie prédite du post
        user_interests: Liste des catégories favorites de l'utilisateur
        created_at: Date de création du post
        user_prefs: Préférences utilisateur (mode, seuil toxicité, etc.)
    
    Returns:
        dict avec score final et détail par composante
    """
    mode = user_prefs.get('mode', 'default')
    w = MODE_WEIGHTS.get(mode, MODE_WEIGHTS['default'])
    threshold = user_prefs.get('toxicity_threshold', 0.3)
    
    # Filtre dur : post trop toxique → score nul
    if toxicity_score > threshold:
        return {
            'score': 0.0,
            'filtered': True,
            'reason': 'toxicity',
            'detail': {
                'similarity': round(float(cosine_sim), 4),
                'quality': round(1.0 - toxicity_score, 4),
                'preference': 0.0,
                'recency': 0.0,
                'mode': mode
            }
        }
    
    # Composante qualité (inverse toxicité)
    quality = 1.0 - toxicity_score
    
    # Composante préférence (bonus si catégorie dans les intérêts)
    # Make preference more extreme for better spread
    if category in user_interests:
        pref = 1.0  # Strong positive for liked categories
    elif user_interests:
        pref = 0.2  # Lower for unknown categories (was 0.3)
    else:
        pref = 0.5  # Neutral when no preferences
    
    # Composante récence (décroissance plus agressive)
    if created_at:
        age_days = (datetime.now() - created_at).days
        # Exponential decay with 14-day half-life (more aggressive)
        recency = max(0.0, math.exp(-age_days / 14.0))
    else:
        recency = 0.3  # Lower default for unknown dates
    
    # Score final pondéré
    raw_score = (
        w['sim'] * max(0, cosine_sim) +
        w['qual'] * quality +
        w['pref'] * pref +
        w['rec'] * recency
    )
    
    # Apply score spreading to get better distribution
    # Use sigmoid spreading for more natural distribution
    final_score = _spread_scores_sigmoid(raw_score, scale=10.0)
    
    return {
        'score': round(float(final_score), 6),
        'filtered': False,
        'detail': {
            'similarity': round(float(cosine_sim), 4),
            'quality': round(quality, 4),
            'preference': round(pref, 4),
            'recency': round(recency, 4),
            'mode': mode,
            'raw_score': round(raw_score, 4)  # Include raw for debugging
        }
    }


def compute_score_with_ranking(
    cosine_sim: float,
    toxicity_score: float,
    category: str,
    user_interests: list[str],
    created_at: datetime | None,
    user_prefs: dict,
    rank_position: int = None,
    total_candidates: int = None
) -> dict:
    """
    Version améliorée avec prise en compte du ranking pour meilleure distribution.
    """
    result = compute_score(
        cosine_sim, toxicity_score, category, 
        user_interests, created_at, user_prefs
    )
    
    # Apply rank-based boost if position provided
    if rank_position is not None and total_candidates is not None and total_candidates > 0:
        # Higher rank (lower position number) gets a boost
        rank_boost = 1.0 - (rank_position / total_candidates)
        rank_boost = 0.8 + (rank_boost * 0.2)  # 0.8-1.0 boost factor
        
        result['score'] = min(0.99, result['score'] * rank_boost)
        result['detail']['rank_boost'] = round(rank_boost, 4)
    
    return result