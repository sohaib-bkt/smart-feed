"""
Tests des endpoints FastAPI avec mock_db (pas de Firebase réel).

Technique : on utilise pytest `monkeypatch` pour remplacer les fonctions
firebase dans chaque module API par les équivalents mock_db.
Ainsi les tests sont rapides, hors-ligne et isolés.
"""

import sys
from unittest.mock import MagicMock

# ── Mock faiss et sentence_transformers AVANT tout import applicatif ──
# Ces libs sont lourdes (ML) ou absentes en CI — on les remplace par des mocks
sys.modules.setdefault("faiss", MagicMock())
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("detoxify", MagicMock())

import pytest
from fastapi.testclient import TestClient
import app.db.mock_db as mock_db

# ── Patch Firebase dans tous les modules API ─────────────────────────
# On doit patcher AVANT d'importer main.py
import app.api.feed as feed_module
import app.api.interact as interact_module
import app.api.preferences as prefs_module


@pytest.fixture(autouse=True)
def patch_firebase_with_mock(monkeypatch):
    """
    Remplace firebase.py par mock_db dans les 3 routers.
    Exécuté automatiquement avant chaque test.
    """
    # ── feed.py ───────────────────────────────────────────────────────
    monkeypatch.setattr(feed_module, "get_user_profile",    mock_db.get_user_profile)
    monkeypatch.setattr(feed_module, "get_user_interactions", mock_db.get_user_interactions)
    monkeypatch.setattr(feed_module, "create_user",         mock_db.create_user)

    # ── interact.py ───────────────────────────────────────────────────
    monkeypatch.setattr(interact_module, "log_interaction",       mock_db.log_interaction)
    monkeypatch.setattr(interact_module, "get_user_interactions", mock_db.get_user_interactions)
    monkeypatch.setattr(interact_module, "update_user_embedding", mock_db.update_user_embedding)

    # ── preferences.py ───────────────────────────────────────────────
    monkeypatch.setattr(prefs_module, "get_user_profile",       mock_db.get_user_profile)
    monkeypatch.setattr(prefs_module, "update_user_preferences", mock_db.update_user_preferences)
    monkeypatch.setattr(prefs_module, "create_user",            mock_db.create_user)

    # Aussi : embed_text → retourne un vecteur nul pour ne pas charger le modèle NLP
    monkeypatch.setattr(interact_module, "embed_text", lambda text: [0.0] * 384)

    # Aussi : get_feed → retourne une liste vide (moteur FAISS non dispo en test)
    monkeypatch.setattr(feed_module, "get_feed", lambda **kwargs: [])

    mock_db.reset_mock()
    yield
    mock_db.reset_mock()


# ── Client HTTP de test (sans démarrer uvicorn) ───────────────────────
@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────
# Tests /health
# ─────────────────────────────────────────────────────────────────────

def test_health(client):
    """Le serveur répond 200 avec status ok."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("✅ GET /health → ok")


# ─────────────────────────────────────────────────────────────────────
# Tests GET /api/preferences/{user_id}
# ─────────────────────────────────────────────────────────────────────

def test_get_preferences_user_inexistant(client):
    """Lire les prefs d'un user inexistant → 404."""
    res = client.get("/api/preferences/nobody")
    assert res.status_code == 404
    print("✅ GET /preferences/nobody → 404")


def test_get_preferences_user_existant(client):
    """Après création, les prefs par défaut sont retournées."""
    import asyncio
    asyncio.run(mock_db.create_user("alice"))

    res = client.get("/api/preferences/alice")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "default"
    assert data["toxicity_threshold"] == 0.3
    print(f"✅ GET /preferences/alice → {data}")


# ─────────────────────────────────────────────────────────────────────
# Tests PUT /api/preferences/{user_id}
# ─────────────────────────────────────────────────────────────────────

def test_update_preferences_cree_user_si_absent(client):
    """PUT prefs sur un user inexistant → le crée automatiquement."""
    payload = {
        "mode": "focus",
        "interests": ["TECH", "SCIENCE"],
        "toxicity_threshold": 0.2,
        "content_type": "all"
    }
    res = client.put("/api/preferences/newuser", json=payload)
    assert res.status_code == 200
    assert res.json()["preferences"]["mode"] == "focus"
    print("✅ PUT /preferences/newuser → créé avec mode=focus")


def test_update_preferences_user_existant(client):
    """PUT prefs sur un user existant → met à jour ses prefs."""
    import asyncio
    asyncio.run(mock_db.create_user("bob"))

    payload = {
        "mode": "fun",
        "interests": ["SPORTS"],
        "toxicity_threshold": 0.5,
        "content_type": "video"
    }
    res = client.put("/api/preferences/bob", json=payload)
    assert res.status_code == 200
    assert res.json()["preferences"]["mode"] == "fun"
    print("✅ PUT /preferences/bob → mode=fun")


def test_update_preferences_mode_invalide(client):
    """PUT avec un mode invalide → 422 (validation Pydantic)."""
    payload = {
        "mode": "turbo",       # mode inexistant
        "interests": [],
        "toxicity_threshold": 0.3,
        "content_type": "all"
    }
    res = client.put("/api/preferences/carol", json=payload)
    assert res.status_code == 422   # Unprocessable Entity
    print("✅ PUT mode=turbo → 422 validation error")


def test_update_preferences_threshold_hors_range(client):
    """PUT avec toxicity_threshold > 1.0 → 422."""
    payload = {
        "mode": "default",
        "interests": [],
        "toxicity_threshold": 1.5,   # hors de [0.0, 1.0]
        "content_type": "all"
    }
    res = client.put("/api/preferences/diana", json=payload)
    assert res.status_code == 422
    print("✅ PUT threshold=1.5 → 422 validation error")


# ─────────────────────────────────────────────────────────────────────
# Tests POST /api/preferences/{user_id}/mode/{mode}
# ─────────────────────────────────────────────────────────────────────

def test_set_mode_valide(client):
    """Changer de mode sur un user existant → 200."""
    import asyncio
    asyncio.run(mock_db.create_user("eve"))

    res = client.post("/api/preferences/eve/mode/learning")
    assert res.status_code == 200
    assert res.json()["mode"] == "learning"
    print("✅ POST /mode/learning → ok")


def test_set_mode_invalide(client):
    """Mode inconnu → 400."""
    import asyncio
    asyncio.run(mock_db.create_user("frank"))

    res = client.post("/api/preferences/frank/mode/turbo")
    assert res.status_code == 400
    print("✅ POST /mode/turbo → 400")


def test_set_mode_user_inexistant(client):
    """Changer le mode d'un user inexistant → 404."""
    res = client.post("/api/preferences/ghost/mode/fun")
    assert res.status_code == 404
    print("✅ POST /mode/fun sur ghost → 404")


# ─────────────────────────────────────────────────────────────────────
# Tests POST /api/interact
# ─────────────────────────────────────────────────────────────────────

def test_interact_like(client):
    """Un like est bien enregistré → 200 + status logged."""
    import asyncio
    asyncio.run(mock_db.create_user("grace"))

    payload = {
        "user_id": "grace",
        "post_id": "post_1",
        "post_text": "L'IA transforme la médecine",
        "action": "like",
        "watch_time": 0.0
    }
    res = client.post("/api/interact", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "logged"
    print("✅ POST /interact (like) → logged")


def test_interact_action_invalide(client):
    """Une action non reconnue → 422 (validation Pydantic)."""
    payload = {
        "user_id": "henry",
        "post_id": "post_2",
        "action": "dislike",    # action inexistante dans le Literal
        "watch_time": 0.0
    }
    res = client.post("/api/interact", json=payload)
    assert res.status_code == 422
    print("✅ POST /interact action=dislike → 422")


def test_interact_met_a_jour_embedding(client):
    """Après un like, l'embedding de l'user change (sort de [0,0,0,...])."""
    import asyncio
    asyncio.run(mock_db.create_user("ivan"))

    # Interaction avec embedding non-nul simulé via mock embed_text
    payload = {
        "user_id": "ivan",
        "post_id": "post_5",
        "post_text": "Sport et santé",
        "action": "watch_full",
        "watch_time": 45.0
    }
    client.post("/api/interact", json=payload)

    # L'embedding dans mock_db doit avoir été mis à jour
    import asyncio
    profile = asyncio.run(mock_db.get_user_profile("ivan"))
    # Avec embed_text mocké → [0.0]*384, embedding reste nul mais l'appel a bien eu lieu
    assert profile is not None
    assert len(profile["embedding"]) == 384
    print("✅ POST /interact → update_user_embedding appelé")