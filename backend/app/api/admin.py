"""
Endpoints d'administration 

POST /admin/retrain       -> ré-entraîne LightFM + XGBoost en arrière-plan
POST /admin/rebuild-index -> reconstruit l'index FAISS v2 en arrière-plan

Ces endpoints déclenchent des scripts Python lourds via subprocess
sans bloquer le serveur (BackgroundTasks FastAPI).
"""

import subprocess
import logging
from fastapi import APIRouter, BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter()

# Séquence complète de ré-entraînement Phase 2
_RETRAIN_STEPS = [
    ["python", "scripts/phase2/build_interaction_matrix.py"],
    ["python", "scripts/phase2/train_lightfm.py"],
    ["python", "scripts/phase2/build_xgb_dataset.py"],
    ["python", "scripts/phase2/train_xgboost.py"],
]

# Script de reconstruction de l'index FAISS v2
_REBUILD_INDEX_SCRIPT = ["python", "scripts/phase2/build_faiss_v2.py"]


def _run_retrain() -> None:
    """
    Lance la pipeline de ré-entraînement complète en arrière-plan.
    S'arrête au premier script en erreur et log le stderr.
    """
    logger.info("Retrain démarré.")
    for cmd in _RETRAIN_STEPS:
        logger.info("Lancement : %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "Erreur sur '%s' (code %d) :\n%s",
                " ".join(cmd),
                result.returncode,
                result.stderr,
            )
            return  # on arrête la pipeline en cas d'erreur
        logger.info("OK : %s", " ".join(cmd))
    logger.info("Retrain terminé avec succès.")


def _run_rebuild_index() -> None:
    """
    Reconstruit l'index FAISS v2 en arrière-plan.
    """
    logger.info("Reconstruction de l'index FAISS v2 démarrée.")
    result = subprocess.run(_REBUILD_INDEX_SCRIPT, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Erreur build_faiss_v2 (code %d) :\n%s",
            result.returncode,
            result.stderr,
        )
        return
    logger.info("Index FAISS v2 reconstruit avec succès.")


@router.post(
    "/admin/retrain",
    summary="Ré-entraîne LightFM + XGBoost (Phase 2)",
    tags=["admin"],
)
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Déclenche le pipeline complet de ré-entraînement en arrière-plan :

    1. `build_interaction_matrix.py`  → matrice interactions Firestore
    2. `train_lightfm.py`             → modèle collaboratif
    3. `build_xgb_dataset.py`         → dataset features pour XGBoost
    4. `train_xgboost.py`             → modèle de re-ranking

    Retourne immédiatement — le retrain tourne en tâche de fond.
    Durée estimée : **5 à 10 minutes**.
    """
    background_tasks.add_task(_run_retrain)
    return {
        "status": "retrain_started",
        "estimated_time": "5-10min",
        "steps": [" ".join(s) for s in _RETRAIN_STEPS],
    }


@router.post(
    "/admin/rebuild-index",
    summary="Reconstruit l'index FAISS v2",
    tags=["admin"],
)
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Reconstruit l'index FAISS v2 à partir des embeddings Phase 2.

    À appeler après avoir ajouté de nouveaux articles dans Firestore.
    Retourne immédiatement — la reconstruction tourne en tâche de fond.
    """
    background_tasks.add_task(_run_rebuild_index)
    return {
        "status": "rebuild_started",
        "script": " ".join(_REBUILD_INDEX_SCRIPT),
    }