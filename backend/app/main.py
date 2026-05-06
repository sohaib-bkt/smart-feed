"""
Point d'entrée de l'API Smart Feed.

Lancer :   uvicorn app.main:app --reload --port 8000
Swagger :  http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import feed, interact, preferences

app = FastAPI(
    title="Smart Feed API",
    description="Système de navigation intelligente de feed — Phase 1",
    version="1.0.0",
)

# ── CORS — autorise l'app Expo (mobile) à appeler l'API ──────────────
# En production, remplacer allow_origins=['*'] par l'URL exacte de l'app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(feed.router,        prefix="/api", tags=["feed"])
app.include_router(interact.router,    prefix="/api", tags=["interactions"])
app.include_router(preferences.router, prefix="/api", tags=["preferences"])


# ── Health check ──────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    """Vérifie que le serveur est bien démarré."""
    return {"status": "ok", "version": "1.0.0"}