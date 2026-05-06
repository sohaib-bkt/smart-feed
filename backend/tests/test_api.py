"""
Tests des endpoints FastAPI avec mock_db (pas de Firebase réel).

Technique : on mock les modules lourds (faiss, numpy, sentence_transformers)
AVANT tout import applicatif pour éviter les dépendances manquantes.
"""

import sys
from unittest.mock import MagicMock, patch
import importlib

# Ces modules sont lourds ou absents → on les remplace par des mocks légers

# Créer un faux module numpy avec juste ce qu'il faut
import types

# Mock numpy
if 'numpy' not in sys.modules:
    mock_np = MagicMock()
    mock_np.array = lambda x, **kwargs: x
    mock_np.ndarray = list
    mock_np.float32 = float
    mock_np.sum = sum
    mock_np.dot = lambda a, b: 0.0
    mock_np.linalg = MagicMock()
    mock_np.linalg.norm = lambda x: 1.0
    mock_np.load = MagicMock(return_value=MagicMock())
    mock_np.save = MagicMock()
    sys.modules['numpy'] = mock_np

# Mock pandas
if 'pandas' not in sys.modules:
    mock_pd = MagicMock()
    mock_pd.read_parquet = MagicMock(return_value=MagicMock())
    mock_pd.DataFrame = MagicMock()
    mock_pd.Timestamp = MagicMock()
    mock_pd.notna = lambda x: True
    sys.modules['pandas'] = mock_pd

# Mock sentence_transformers
if 'sentence_transformers' not in sys.modules:
    mock_st = MagicMock()
    mock_st.SentenceTransformer = MagicMock()
    sys.modules['sentence_transformers'] = mock_st

# Mock transformers
if 'transformers' not in sys.modules:
    sys.modules['transformers'] = MagicMock()

# Mock detoxify
if 'detoxify' not in sys.modules:
    sys.modules['detoxify'] = MagicMock()

# Mock faiss
if 'faiss' not in sys.modules:
    mock_faiss = MagicMock()
    mock_faiss.IndexFlatIP = MagicMock()
    mock_faiss.read_index = MagicMock(return_value=MagicMock())
    mock_faiss.write_index = MagicMock()
    sys.modules['faiss'] = mock_faiss

# Mock firebase_admin
if 'firebase_admin' not in sys.modules:
    sys.modules['firebase_admin'] = MagicMock()

# Mock datasets (HuggingFace)
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
    Exécuté automatiquement avant chaque test.
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
    """Le serveur répond 200 avec status ok."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("✅ GET /health → ok")


# ═════════════════════════════════════════════════════════════════════
# Tests GET /api/preferences/{user_id}
# ═════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════
# Tests PUT /api/preferences/{user_id}
# ═════════════════════════════════════════════════════════════════════

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
        "mode": "turbo",
        "interests": [],
        "toxicity_threshold": 0.3,
        "content_type": "all"
    }
    res = client.put("/api/preferences/carol", json=payload)
    assert res.status_code == 422
    print("✅ PUT mode=turbo → 422 validation error")


def test_update_preferences_threshold_hors_range(client):
    """PUT avec toxicity_threshold > 1.0 → 422."""
    payload = {
        "mode": "default",
        "interests": [],
        "toxicity_threshold": 1.5,
        "content_type": "all"
    }
    res = client.put("/api/preferences/diana", json=payload)
    assert res.status_code == 422
    print("✅ PUT threshold=1.5 → 422 validation error")


# ═════════════════════════════════════════════════════════════════════
# Tests POST /api/preferences/{user_id}/mode/{mode}
# ═════════════════════════════════════════════════════════════════════

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
    """Une action non reconnue → 422 (validation Pydantic)."""
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