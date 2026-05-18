"""
Tests des endpoints Module 8 (API v2) :
  - GET  /api/feed/{user_id}?version=v2  → pipeline hybride + re-ranking
  - POST /api/admin/retrain              → déclenche pipeline en arrière-plan
  - POST /api/admin/rebuild-index        → reconstruit l'index FAISS

Stratégie : mêmes mocks que test_api.py + mock de get_hybrid_feed
et rank_candidates pour isoler complètement le Module 8.
"""

import sys
from unittest.mock import MagicMock, patch

# Mocks lourds AVANT tout import 
for mod in ["sentence_transformers", "transformers", "detoxify", "datasets"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

if "faiss" not in sys.modules:
    mock_faiss = MagicMock()
    mock_faiss.read_index = MagicMock(return_value=MagicMock())
    sys.modules["faiss"] = mock_faiss

if "firebase_admin" not in sys.modules:
    sys.modules["firebase_admin"] = MagicMock()

if "apscheduler" not in sys.modules:
    mock_aps = MagicMock()
    mock_aps.schedulers = MagicMock()
    mock_aps.schedulers.background = MagicMock()
    mock_scheduler_instance = MagicMock()
    mock_aps.schedulers.background.BackgroundScheduler = MagicMock(
        return_value=mock_scheduler_instance
    )
    sys.modules["apscheduler"] = mock_aps
    sys.modules["apscheduler.schedulers"] = mock_aps.schedulers
    sys.modules["apscheduler.schedulers.background"] = mock_aps.schedulers.background

# Imports réels 
import pytest
from fastapi.testclient import TestClient
import app.db.mock_db as mock_db
import app.api.feed as feed_module
import app.api.admin as admin_module

# Candidats fictifs retournés par get_hybrid_feed
FAKE_CANDIDATES = [
    {
        "id": f"post_{i}",
        "headline": f"Article test {i}",
        "category": "TECH",
        "score": round(0.8 - i * 0.05, 2),
        "score_detail": {"content_based": 0.7, "collaborative": 0.6, "cosine_sim": 0.65},
        "explanation": "Recommandé pour vous",
        "toxicity_score": 0.05 * i,
        "likes": 100 * (5 - i),
        "views": 500 * (5 - i),
    }
    for i in range(5)
]


# Fixtures 

@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    """
    Remplace Firebase et les moteurs lourds par des mocks légers.
    Suit exactement le même pattern que test_api.py.
    """
    # Firebase → mock_db
    monkeypatch.setattr(feed_module, "get_user_profile",     mock_db.get_user_profile)
    monkeypatch.setattr(feed_module, "get_user_interactions", mock_db.get_user_interactions)
    monkeypatch.setattr(feed_module, "create_user",          mock_db.create_user)

    # v1 fallback
    monkeypatch.setattr(feed_module, "get_feed", lambda **kwargs: [])

    # v2 — moteur hybride mocké
    monkeypatch.setattr(feed_module, "get_hybrid_feed",
                        lambda **kwargs: list(FAKE_CANDIDATES))

    # rank_candidates — retourne les candidats avec rank_score ajouté
    def fake_rank(candidates, user_embedding=None, weights=None):
        ranked = [dict(c, rank_score=c["score"]) for c in candidates]
        ranked.sort(key=lambda x: x["rank_score"], reverse=True)
        return ranked

    monkeypatch.setattr(feed_module, "rank_candidates", fake_rank)

    mock_db.reset_mock()
    yield
    mock_db.reset_mock()


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════
# Tests GET /api/feed/{user_id}?version=v2
# ═══════════════════════════════════════════════════════════════════════

class TestFeedV2:

    def test_feed_v2_status_200(self, client):
        """Le feed v2 retourne bien un 200."""
        import asyncio
        asyncio.run(mock_db.create_user("alice"))
        res = client.get("/api/feed/alice?version=v2")
        assert res.status_code == 200
        print("✅ GET /feed/alice?version=v2 → 200")

    def test_feed_v2_champs_obligatoires(self, client):
        """La réponse contient tous les champs du contrat API."""
        import asyncio
        asyncio.run(mock_db.create_user("bob"))
        res = client.get("/api/feed/bob?version=v2")
        data = res.json()
        for champ in ("user_id", "version", "ranking_method", "mode", "count", "feed"):
            assert champ in data, f"Champ manquant : {champ}"
        print(f"✅ Champs obligatoires présents : {list(data.keys())}")

    def test_feed_v2_ranking_method_correct(self, client):
        """v2 retourne ranking_method='hybrid_xgboost'."""
        import asyncio
        asyncio.run(mock_db.create_user("carol"))
        res = client.get("/api/feed/carol?version=v2")
        assert res.json()["ranking_method"] == "hybrid_xgboost"
        print("✅ ranking_method=hybrid_xgboost")

    def test_feed_v1_ranking_method_correct(self, client):
        """v1 retourne ranking_method='content_based_v1'."""
        import asyncio
        asyncio.run(mock_db.create_user("diana"))
        res = client.get("/api/feed/diana?version=v1")
        assert res.json()["ranking_method"] == "content_based_v1"
        print("✅ ranking_method=content_based_v1")

    def test_feed_v2_cree_user_si_absent(self, client):
        """Un user inconnu est créé automatiquement en v2."""
        res = client.get("/api/feed/newuser42?version=v2")
        assert res.status_code == 200
        assert res.json()["user_id"] == "newuser42"
        print("✅ User inconnu créé automatiquement")

    def test_feed_v2_respect_limit(self, client):
        """Le paramètre limit est respecté."""
        import asyncio
        asyncio.run(mock_db.create_user("eve"))
        res = client.get("/api/feed/eve?version=v2&limit=3")
        data = res.json()
        assert data["count"] <= 3
        assert len(data["feed"]) <= 3
        print(f"✅ limit=3 respecté → {data['count']} items")

    def test_feed_v2_limit_max_50(self, client):
        """limit > 50 → erreur 422."""
        res = client.get("/api/feed/frank?version=v2&limit=99")
        assert res.status_code == 422
        print("✅ limit=99 → 422")

    def test_feed_v2_version_invalide(self, client):
        """version=v3 → erreur 422 (pattern ^v[12]$)."""
        res = client.get("/api/feed/grace?version=v3")
        assert res.status_code == 422
        print("✅ version=v3 → 422")

    def test_feed_v2_items_ont_rank_score(self, client):
        """Chaque item du feed v2 doit avoir un champ rank_score."""
        import asyncio
        asyncio.run(mock_db.create_user("henry"))
        res = client.get("/api/feed/henry?version=v2&limit=5")
        feed = res.json()["feed"]
        if feed:
            for item in feed:
                assert "rank_score" in item, f"rank_score manquant dans {item.get('id')}"
        print(f"✅ rank_score présent sur {len(feed)} items")

    def test_feed_v2_feed_vide_si_no_candidates(self, client, monkeypatch):
        """Si get_hybrid_feed retourne [], le feed est vide."""
        monkeypatch.setattr(feed_module, "get_hybrid_feed", lambda **kwargs: [])
        import asyncio
        asyncio.run(mock_db.create_user("ivan"))
        res = client.get("/api/feed/ivan?version=v2")
        data = res.json()
        assert data["count"] == 0
        assert data["feed"] == []
        print("✅ Feed vide si pas de candidats")


# ═══════════════════════════════════════════════════════════════════════
# Tests POST /api/admin/retrain
# ═══════════════════════════════════════════════════════════════════════

class TestAdminRetrain:

    def test_retrain_status_200(self, client):
        """POST /admin/retrain retourne 200."""
        res = client.post("/api/admin/retrain")
        assert res.status_code == 200
        print("✅ POST /admin/retrain → 200")

    def test_retrain_status_field(self, client):
        """La réponse contient status='retrain_started'."""
        res = client.post("/api/admin/retrain")
        assert res.json()["status"] == "retrain_started"
        print("✅ status=retrain_started")

    def test_retrain_contient_estimated_time(self, client):
        """La réponse indique le temps estimé."""
        res = client.post("/api/admin/retrain")
        assert "estimated_time" in res.json()
        print("✅ estimated_time présent")

    def test_retrain_contient_steps(self, client):
        """La réponse liste les étapes du pipeline."""
        res = client.post("/api/admin/retrain")
        data = res.json()
        assert "steps" in data
        assert len(data["steps"]) == 4
        print(f"✅ {len(data['steps'])} étapes listées")

    def test_retrain_background_ne_bloque_pas(self, client):
        """L'endpoint retourne immédiatement (pas de timeout)."""
        import time
        start = time.time()
        res = client.post("/api/admin/retrain")
        elapsed = time.time() - start
        assert res.status_code == 200
        assert elapsed < 2.0, f"Réponse trop lente : {elapsed:.2f}s"
        print(f"✅ Réponse en {elapsed:.3f}s (non bloquant)")


# ═══════════════════════════════════════════════════════════════════════
# Tests POST /api/admin/rebuild-index
# ═══════════════════════════════════════════════════════════════════════

class TestAdminRebuildIndex:

    def test_rebuild_status_200(self, client):
        """POST /admin/rebuild-index retourne 200."""
        res = client.post("/api/admin/rebuild-index")
        assert res.status_code == 200
        print("✅ POST /admin/rebuild-index → 200")

    def test_rebuild_status_field(self, client):
        """La réponse contient status='rebuild_started'."""
        res = client.post("/api/admin/rebuild-index")
        assert res.json()["status"] == "rebuild_started"
        print("✅ status=rebuild_started")

    def test_rebuild_contient_script(self, client):
        """La réponse mentionne le script exécuté."""
        res = client.post("/api/admin/rebuild-index")
        data = res.json()
        assert "script" in data
        assert "build_faiss_v2" in data["script"]
        print(f"✅ script={data['script']}")

    def test_rebuild_background_ne_bloque_pas(self, client):
        """L'endpoint retourne immédiatement."""
        import time
        start = time.time()
        res = client.post("/api/admin/rebuild-index")
        elapsed = time.time() - start
        assert res.status_code == 200
        assert elapsed < 2.0
        print(f"✅ Réponse en {elapsed:.3f}s (non bloquant)")


# ═══════════════════════════════════════════════════════════════════════
# Tests /health — vérification version 2.0.0
# ═══════════════════════════════════════════════════════════════════════

class TestHealthV2:

    def test_health_version_v2(self, client):
        """Le health check retourne bien la version 2.0.0."""
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["version"] == "2.0.0"
        print("✅ /health → version=2.0.0")