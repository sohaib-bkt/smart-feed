"""
Endpoints pour gérer les préférences utilisateur.

GET  /api/preferences/{user_id}       : lire les prefs
PUT  /api/preferences/{user_id}          : remplacer les prefs
POST /api/preferences/{user_id}/mode/{mode} : changer juste le mode
"""

from fastapi import APIRouter, HTTPException
from app.schemas.user import UserPreferences
from app.db.firebase import (
    get_user_profile,
    update_user_preferences,
    create_user,
)

router = APIRouter()

VALID_MODES = ["default", "focus", "fun", "learning"]


@router.get(
    "/preferences/{user_id}",
    summary="Lire les préférences d'un utilisateur",
)
async def get_preferences(user_id: str):
    """
    Retourne les préférences actuelles de l'utilisateur.

    **Firestore lit :** `users/{user_id}`

    **Retourne 404** si l'utilisateur n'existe pas encore.
    """
    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Utilisateur '{user_id}' introuvable")
    return profile.get("preferences", {})


@router.put(
    "/preferences/{user_id}",
    summary="Mettre à jour toutes les préférences",
)
async def update_preferences(user_id: str, prefs: UserPreferences):
    """
    Remplace toutes les préférences de l'utilisateur.
    Si l'utilisateur n'existe pas → le crée avec ces préférences.

    **Firestore écrit :** `users/{user_id}/preferences`

    **Exemple de payload :**
    ```json
    {
        "mode": "focus",
        "interests": ["TECH", "SCIENCE"],
        "toxicity_threshold": 0.2,
        "content_type": "all"
    }
    ```
    """
    profile = await get_user_profile(user_id)

    if not profile:
        # Création automatique si l'user n'existe pas encore
        await create_user(user_id, prefs.model_dump())
    else:
        await update_user_preferences(user_id, prefs.model_dump())
    # → Firestore : met à jour users/{user_id}/preferences

    return {"status": "updated", "preferences": prefs.model_dump()}


@router.post(
    "/preferences/{user_id}/mode/{mode}",
    summary="Changer uniquement le mode de navigation",
)
async def set_mode(user_id: str, mode: str):
    """
    Raccourci pour changer de mode sans toucher aux autres préférences.

    **Modes disponibles :** `default` | `focus` | `fun` | `learning`

    **Firestore écrit :** `users/{user_id}/preferences.mode`
    """
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Mode invalide. Valeurs acceptées : {VALID_MODES}",
        )

    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Utilisateur '{user_id}' introuvable")

    prefs = profile.get("preferences", {})
    prefs["mode"] = mode
    await update_user_preferences(user_id, prefs)

    return {"status": "mode_updated", "mode": mode}