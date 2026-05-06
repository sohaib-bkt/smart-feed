"""
tests/test_api.py
──────────────────
Tests des endpoints FastAPI avec mock_db.
"""

import sys
from unittest.mock import MagicMock, patch
import importlib

# ── ÉTAPE CRITIQUE : Mock AVANT tout import ──────────────────────────
# On ne mock QUE les modules qui ne sont PAS installés
# Si numpy/pandas sont installés (via pip), on les laisse tranquilles

# Mock sentence_transformers (lourd, pas nécessaire pour les tests API)
if 'sentence_transformers' not in sys.modules:
    mock_st = MagicMock()
    mock_st.SentenceTransformer = MagicMock()
    sys.modules['sentence_transformers'] = mock_st

# Mock transformers (lourd, pas nécessaire pour les tests API)
if 'transformers' not in sys.modules:
    sys.modules['transformers'] = MagicMock()

# Mock detoxify (pas nécessaire pour les tests API)
if 'detoxify' not in sys.modules:
    sys.modules['detoxify'] = MagicMock()

# Mock faiss (pas nécessaire pour les tests API, on mock get_feed)
if 'faiss' not in sys.modules:
    mock_faiss = MagicMock()
    mock_faiss.IndexFlatIP = MagicMock()
    mock_faiss.read_index = MagicMock(return_value=MagicMock())
    mock_faiss.write_index = MagicMock()
    sys.modules['faiss'] = mock_faiss

# Mock firebase_admin (on utilise mock_db à la place)
if 'firebase_admin' not in sys.modules:
    sys.modules['firebase_admin'] = MagicMock()

# Mock datasets (HuggingFace, pas nécessaire)
if 'datasets' not in sys.modules:
    sys.modules['datasets'] = MagicMock()

# ── Maintenant on peut importer l'app ────────────────────────────────
import pytest
from fastapi.testclient import TestClient
import app.db.mock_db as mock_db

# Patch Firebase dans les modules API
import app.api.feed as feed_module
import app.api.interact as interact_module
import app.api.preferences as prefs_module


@pytest.fixture(autouse=True)
def patch_firebase_with_mock(monkeypatch):
    """
    Remplace firebase.py par mock_db dans les 3 routers.
    """
    # ── feed.py ───────────────────────────────────────────────────────
    monkeypatch.setattr(feed_module, "get_user_profile",    mock_db.get_user_profile)
    monkeypatch.setattr(feed_module, "get_user_interactions", mock_db.get_user_interactions)
    monkeypatch.setattr(feed_module, "create_user",         mock_db.create_user)
    monkeypatch.setattr(feed_module, "get_feed",            lambda **kwargs: [])

    # ── interact.py ───────────────────────────────────────────────────
    monkeypatch.setattr(interact_module, "log_interaction",       mock_db.log_interaction)
    monkeypatch.setattr(interact_module, "get_user_interactions", mock_db.get_user_interactions)
    monkeypatch.setattr(interact_module, "update_user_embedding", mock_db.update_user_embedding)
    monkeypatch.setattr(interact_module, "embed_text",           lambda text: [0.0] * 384)

    # ── preferences.py ───────────────────────────────────────────────
    monkeypatch.setattr(prefs_module, "get_user_profile",       mock_db.get_user_profile)
    monkeypatch.setattr(prefs_module, "update_user_preferences", mock_db.update_user_preferences)
    monkeypatch.setattr(prefs_module, "create_user",            mock_db.create_user)

    mock_db.reset_mock()
    yield
    mock_db.reset_mock()


# ── Client HTTP de test ──────────────────────────────────────────────
@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ═════════════════════════════════════════════════════════════════════
# Tests /health
# ═════════════════════════════════════════════════════════════════════

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("✅ GET /health → ok")


# ═════════════════════════════════════════════════════════════════════
# Tests GET /api/preferences/{user_id}
# ═════════════════════════════════════════════════════════════════════

def test_get_preferences_user_inexistant(client):
    res = client.get("/api/preferences/nobody")
    assert res.status_code == 404
    print("✅ GET /preferences/nobody → 404")


def test_get_preferences_user_existant(client):
    import asyncio
    asyncio.run(mock_db.create_user("alice"))
    res = client.get("/api/preferences/alice")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "default"
    print(f"✅ GET /preferences/alice → {data}")


# ═════════════════════════════════════════════════════════════════════
# Tests PUT /api/preferences/{user_id}
# ═════════════════════════════════════════════════════════════════════

def test_update_preferences_cree_user_si_absent(client):
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
    payload = {
        "mode": "turbo",
        "interests": [],
        "toxicity_threshold": 0.3,
        "content_type": "all"
    }
    res = client.put("/api/preferences/carol", json=payload)
    assert res.status_code == 422
    print("✅ PUT mode=turbo → 422")


def test_update_preferences_threshold_hors_range(client):
    payload = {
        "mode": "default",
        "interests": [],
        "toxicity_threshold": 1.5,
        "content_type": "all"
    }
    res = client.put("/api/preferences/diana", json=payload)
    assert res.status_code == 422
    print("✅ PUT threshold=1.5 → 422")


# ═════════════════════════════════════════════════════════════════════
# Tests POST /api/preferences/{user_id}/mode/{mode}
# ═════════════════════════════════════════════════════════════════════

def test_set_mode_valide(client):
    import asyncio
    asyncio.run(mock_db.create_user("eve"))
    res = client.post("/api/preferences/eve/mode/learning")
    assert res.status_code == 200
    assert res.json()["mode"] == "learning"
    print("✅ POST /mode/learning → ok")


def test_set_mode_invalide(client):
    import asyncio
    asyncio.run(mock_db.create_user("frank"))
    res = client.post("/api/preferences/frank/mode/turbo")
    assert res.status_code == 400
    print("✅ POST /mode/turbo → 400")


def test_set_mode_user_inexistant(client):
    res = client.post("/api/preferences/ghost/mode/fun")
    assert res.status_code == 404
    print("✅ POST /mode/fun sur ghost → 404")


# ═════════════════════════════════════════════════════════════════════
# Tests POST /api/interact
# ═════════════════════════════════════════════════════════════════════

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
    """Une action non reconnue → 422."""
    payload = {
        "user_id": "henry",
        "post_id": "post_2",
        "action": "dislike",
        "watch_time": 0.0
    }
    res = client.post("/api/interact", json=payload)
    assert res.status_code == 422
    print("✅ POST /interact action=dislike → 422")


def test_interact_met_a_jour_embedding(client):
    """Après un like, l'embedding de l'user est mis à jour."""
    import asyncio
    asyncio.run(mock_db.create_user("ivan"))
    payload = {
        "user_id": "ivan",
        "post_id": "post_5",
        "post_text": "Sport et santé",
        "action": "watch_full",
        "watch_time": 45.0
    }
    client.post("/api/interact", json=payload)
    profile = asyncio.run(mock_db.get_user_profile("ivan"))
    assert profile is not None
    assert len(profile["embedding"]) == 384
    print("✅ POST /interact → update_user_embedding appelé")