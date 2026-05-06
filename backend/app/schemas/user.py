"""
Schémas Pydantic pour la validation des données utilisateur.
Pydantic vérifie automatiquement les types et valeurs à chaque requête API.
"""

from pydantic import BaseModel, Field
from typing import Literal


class UserPreferences(BaseModel):
    """
    Préférences d'un utilisateur — validées à chaque PUT /preferences/{user_id}.

    Exemple de payload JSON valide :
        {
            "mode": "focus",
            "interests": ["TECH", "SCIENCE"],
            "toxicity_threshold": 0.2,
            "content_type": "all"
        }
    """

    mode: Literal["default", "focus", "fun", "learning"] = "default"

    interests: list[str] = Field(
        default_factory=list,
        description="Catégories favorites (ex: TECH, SPORTS, SCIENCE)",
    )

    toxicity_threshold: float = Field(
        default=0.3,
        ge=0.0,   # >= 0
        le=1.0,   # <= 1
        description="Seuil de filtrage toxicité (0 = strict, 1 = permissif)",
    )

    content_type: Literal["all", "video", "article", "image"] = "all"