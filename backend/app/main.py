"""
Point d'entrée de l'API Smart Feed (Phase 2)

Lancer :   uvicorn app.main:app --reload --port 8000
Swagger :  http://localhost:8000/docs
"""

import subprocess
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from app.api import feed, interact, preferences, admin

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Feed API",
    description="Système de recommandation hybride (content-based + collaboratif) — Phase 2",
    version="2.0.0",
)

# CORS : autorise l'app Expo (mobile) à appeler l'API 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers 
app.include_router(feed.router,        prefix="/api", tags=["feed"])
app.include_router(interact.router,    prefix="/api", tags=["interactions"])
app.include_router(preferences.router, prefix="/api", tags=["preferences"])
app.include_router(admin.router,       prefix="/api", tags=["admin"])


# Ré-entraînement automatique tous les jours à 2h du matin 
def nightly_retrain() -> None:
    """Pipeline de ré-entraînement nocturne LightFM + XGBoost."""
    logger.info("Nightly retrain démarré...")
    for script in [
        "scripts/phase2/build_interaction_matrix.py",
        "scripts/phase2/train_lightfm.py",
        "scripts/phase2/train_xgboost.py",
    ]:
        result = subprocess.run(["python", script], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Erreur nightly retrain sur %s :\n%s", script, result.stderr)
            return
    logger.info("Nightly retrain terminé.")


scheduler = BackgroundScheduler()
scheduler.add_job(nightly_retrain, "cron", hour=2, minute=0)
scheduler.start()


# Health check 
@app.get("/health", tags=["system"])
def health():
    """Vérifie que le serveur est bien démarré."""
    return {"status": "ok", "version": "2.0.0"}