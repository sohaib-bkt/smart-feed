from datetime import datetime
import math

# Poids par mode utilisateur
MODE_WEIGHTS = {
    'default':   {'sim': 0.40, 'qual': 0.30, 'pref': 0.20, 'rec': 0.10},
    'focus':     {'sim': 0.60, 'qual': 0.25, 'pref': 0.10, 'rec': 0.05},
    'fun':       {'sim': 0.20, 'qual': 0.20, 'pref': 0.50, 'rec': 0.10},
    'learning':  {'sim': 0.50, 'qual': 0.30, 'pref': 0.15, 'rec': 0.05},
}

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
    pref = 1.0 if category in user_interests else 0.3
    
    # Composante récence (décroissance linéaire sur 30 jours)
    if created_at:
        age_days = (datetime.now() - created_at).days
        recency = max(0.0, 1.0 - age_days / 30.0)
    else:
        recency = 0.5  # valeur neutre si date inconnue
    
    # Score final pondéré
    score = (
        w['sim']  * max(0, cosine_sim) +
        w['qual'] * quality +
        w['pref'] * pref +
        w['rec']  * recency
    )
    
    return {
        'score': round(float(score), 6),
        'filtered': False,
        'detail': {
            'similarity': round(float(cosine_sim), 4),
            'quality': round(quality, 4),
            'preference': round(pref, 4),
            'recency': round(recency, 4),
            'mode': mode
        }
    }