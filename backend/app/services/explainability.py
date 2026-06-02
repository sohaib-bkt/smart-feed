"""
Couche d'explicabilité du feed Smart Feed : 

Annote chaque post du feed avec une explication lisible par l'utilisateur
final, structurée en 7 dimensions :

  1. Similarité sémantique   (cosine_sim)
  2. Filtrage collaboratif   (cf_score)
  3. Correspondance catégorie (category vs user_prefs.interests)
  4. Récence                 (recency)
  5. Modalité préférée       (modal vs user_prefs.content_type)
  6. Toxicité / santé        (toxicity_score)
  7. Score XGBoost / final   (xgb_score ou score)

"""

from __future__ import annotations

from typing import Any


Post         = dict[str, Any]
UserPrefs    = dict[str, Any]
Interactions = list[dict[str, Any]]


_MODAL_LABELS: dict[str, str] = {
    "text":  "article texte",
    "image": "post image",
    "video": "vidéo courte",
}


# ---------------------------------------------------------------------------
# Fonctions privées une par dimension
# ---------------------------------------------------------------------------

def _reason_similarity(sim: float, detail: dict) -> str | None:
    """Dimension 1 : similarité sémantique cosinus."""
    detail["similarity"] = round(sim, 3)
    if sim > 0.75:
        return f"très similaire à tes intérêts ({sim:.0%})"
    if sim > 0.55:
        return f"similaire à tes intérêts ({sim:.0%})"
    if sim > 0.35:
        return "partiellement lié à tes intérêts"
    return None


def _reason_collaborative(cf: float, detail: dict) -> str | None:
    """Dimension 2 : filtrage collaboratif."""
    detail["collaborative"] = round(cf, 3)
    if cf > 0.72:
        return "très populaire chez des lecteurs similaires"
    if cf > 0.60:
        return "apprécié par des lecteurs similaires"
    return None


def _reason_category(cat: str, prefs: UserPrefs, detail: dict) -> str | None:
    """Dimension 3 : correspondance catégorie / intérêts."""
    interests: list[str] = prefs.get("interests", [])
    match = cat in interests
    detail["category_match"] = match
    if match:
        return f"catégorie {cat} dans tes favoris"
    return None


def _reason_recency(rec: float, detail: dict) -> str | None:
    """Dimension 4 : récence du contenu."""
    detail["recency"] = round(rec, 2)
    if rec > 0.85:
        return "contenu très récent"
    if rec > 0.65:
        return "contenu récent"
    return None


def _reason_modal(modal: str, prefs: UserPrefs, detail: dict) -> str | None:
    """Dimension 5 : modalité préférée de l'utilisateur."""
    detail["modal"] = modal
    pref_type = prefs.get("content_type", "all")
    if pref_type != "all" and modal == pref_type:
        label = _MODAL_LABELS.get(modal, modal)
        return f"format {label} préféré"
    return None


def _reason_toxicity(tox: float, detail: dict) -> str | None:
    """Dimension 6 : qualité / faible toxicité."""
    detail["toxicity"] = round(tox, 3)
    if tox < 0.03:
        return "contenu très sain"
    if tox < 0.10:
        return "faible toxicité"
    return None


def _extract_xgb_score(post: Post) -> float:
    """Résout le score final XGBoost depuis les champs disponibles."""
    raw = post.get("xgb_score") or post.get("rank_score") or post.get("score") or 0.0
    return float(raw)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def build_explanation(
    post: Post,
    user_prefs: UserPrefs,
    interactions: Interactions,
) -> dict[str, Any]:
    """
    Construit l'objet d'explication pour un post donné.

    Args:
        post         : dict du post tel que retourné par le pipeline de
                       recommandation (doit contenir au moins 'score').
        user_prefs   : préférences de l'utilisateur (interests, content_type…).
        interactions : historique d'interactions (non utilisé pour les raisons
                       individuelles, gardé pour extension future).

    Returns:
        dict avec les clés :
          - reasons  : list[str]  — toutes les raisons trouvées
          - summary  : str        — résumé lisible (3 raisons max)
          - detail   : dict       — valeurs brutes par dimension
          - score    : float      — score final arrondi à 4 décimales
    """
    reasons: list[str] = []
    detail:  dict[str, Any] = {}

    # --- Dimension 1 : similarité sémantique ---
    sim = float(post.get("cosine_sim") or 0.0)
    # Fallback : chercher dans score_detail si cosine_sim absent
    if sim == 0.0:
        score_detail = post.get("score_detail") or {}
        sim = float(
            score_detail.get("cosine_sim") or
            score_detail.get("similarity") or 0.0
        )
    r = _reason_similarity(sim, detail)
    if r:
        reasons.append(r)

    # --- Dimension 2 : filtrage collaboratif ---
    cf = float(post.get("cf_score") or 0.5)
    if cf == 0.5:
        cf = float((post.get("score_detail") or {}).get("collaborative") or 0.5)
    r = _reason_collaborative(cf, detail)
    if r:
        reasons.append(r)

    # --- Dimension 3 : catégorie préférée ---
    cat = str(post.get("category") or "")
    r = _reason_category(cat, user_prefs, detail)
    if r:
        reasons.append(r)

    # --- Dimension 4 : récence ---
    rec = float(post.get("recency") or 0.5)
    if rec == 0.5:
        rec = float((post.get("score_detail") or {}).get("recency") or 0.5)
    r = _reason_recency(rec, detail)
    if r:
        reasons.append(r)

    # --- Dimension 5 : modalité ---
    modal = str(post.get("modal") or "text")
    r = _reason_modal(modal, user_prefs, detail)
    if r:
        reasons.append(r)

    # --- Dimension 6 : toxicité / santé ---
    tox = float(post.get("toxicity_score") or 0.0)
    r = _reason_toxicity(tox, detail)
    if r:
        reasons.append(r)

    # --- Dimension 7 : score XGBoost / final ---
    xgb = _extract_xgb_score(post)
    detail["xgb_score"] = round(xgb, 4)

    # --- Fallback si aucune raison trouvée ---
    if not reasons:
        reasons = ["recommandé selon ton profil"]

    summary = "Recommandé car : " + ", ".join(reasons[:3])

    return {
        "reasons": reasons,
        "summary": summary,
        "detail":  detail,
        "score":   round(xgb, 4),
    }


def annotate_feed(
    feed: list[Post],
    user_prefs: UserPrefs,
    interactions: Interactions,
) -> None:
    """
    Annote chaque post du feed **en place** avec une explication.

    Chaque post reçoit un champ ``post['explanation']`` contenant :
      - reasons  : list[str]
      - summary  : str
      - detail   : dict
      - score    : float

    Args:
        feed         : liste de posts (modifiée en place).
        user_prefs   : préférences utilisateur.
        interactions : historique d'interactions de l'utilisateur.
    """
    for post in feed:
        post["explanation"] = build_explanation(post, user_prefs, interactions)