"""
test_ranker.py : Tests unitaires pour app/models/ranker.py

Couvre :
- Chaque sous-score isolément (_similarity, _clean, _popularity, _freshness)
- score_candidate (intégration des 4 sous-scores)
- rank_candidates (tri, champ rank_score, cas limites)
- Cohérence de la formule pondérée (poids somment à 1)
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.models.ranker import (
    RANKER_WEIGHTS,
    _clean_score,
    _freshness_score,
    _popularity_score,
    _similarity_score,
    rank_candidates,
    score_candidate,
)

# Fixtures

DIM = 384


def make_emb(value: float, dim: int = DIM) -> list[float]:
    """Vecteur constant, normalisé à 1."""
    v = [value] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def make_candidate(
    post_emb: list[float] | None = None,
    toxicity: float = 0.0,
    likes: int = 0,
    views: int = 0,
    created_at: datetime | None = None,
    cid: str = "post_1",
) -> dict:
    return {
        "id": cid,
        "embedding": post_emb or make_emb(1.0),
        "toxicity_score": toxicity,
        "likes": likes,
        "views": views,
        "created_at": created_at,
    }


# Tests : _similarity_score

class TestSimilarityScore:

    def test_identical_embeddings_returns_one(self):
        emb = make_emb(1.0)
        assert _similarity_score(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_embeddings_returns_zero(self):
        emb = make_emb(1.0)
        neg = [-x for x in emb]
        assert _similarity_score(emb, neg) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_returns_half(self):
        # Vecteurs orthogonaux → cosine = 0 → score = 0.5
        u = [0.0] * DIM
        u[0] = 1.0
        p = [0.0] * DIM
        p[1] = 1.0
        assert _similarity_score(u, p) == pytest.approx(0.5, abs=1e-5)

    def test_zero_user_embedding_returns_neutral(self):
        post = make_emb(1.0)
        assert _similarity_score([0.0] * DIM, post) == 0.5

    def test_zero_post_embedding_returns_neutral(self):
        user = make_emb(1.0)
        assert _similarity_score(user, [0.0] * DIM) == 0.5

    def test_output_in_zero_one(self):
        u = make_emb(0.5)
        p = make_emb(0.8)
        s = _similarity_score(u, p)
        assert 0.0 <= s <= 1.0


# Tests : _clean_score

class TestCleanScore:

    def test_zero_toxicity_returns_one(self):
        assert _clean_score(0.0) == 1.0

    def test_full_toxicity_returns_zero(self):
        assert _clean_score(1.0) == 0.0

    def test_mid_toxicity(self):
        assert _clean_score(0.5) == pytest.approx(0.5)

    def test_clamp_above_one(self):
        assert _clean_score(1.5) == 0.0

    def test_clamp_below_zero(self):
        assert _clean_score(-0.1) == 1.0

    def test_output_in_zero_one(self):
        for v in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert 0.0 <= _clean_score(v) <= 1.0


# Tests : _popularity_score

class TestPopularityScore:

    def test_zero_engagement_returns_zero(self):
        assert _popularity_score(0, 0) == 0.0

    def test_output_in_zero_one(self):
        assert 0.0 <= _popularity_score(100, 500) <= 1.0

    def test_more_likes_higher_score(self):
        assert _popularity_score(1000, 0) > _popularity_score(10, 0)

    def test_more_views_higher_score(self):
        assert _popularity_score(0, 5000) > _popularity_score(0, 50)

    def test_very_high_engagement_capped_at_one(self):
        assert _popularity_score(10_000_000, 10_000_000) <= 1.0


# ---------------------------------------------------------------------------
# Tests : _freshness_score
# ---------------------------------------------------------------------------

class TestFreshnessScore:

    def test_none_created_at_returns_one(self):
        assert _freshness_score(None) == 1.0

    def test_fresh_post_near_one(self):
        now = datetime.now(timezone.utc)
        score = _freshness_score(now)
        assert score > 0.99

    def test_week_old_post_near_half(self):
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        score = _freshness_score(week_ago)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_old_post_lower_than_fresh(self):
        old   = datetime.now(timezone.utc) - timedelta(days=30)
        fresh = datetime.now(timezone.utc)
        assert _freshness_score(old) < _freshness_score(fresh)

    def test_output_in_zero_one(self):
        old = datetime.now(timezone.utc) - timedelta(days=365)
        assert 0.0 <= _freshness_score(old) <= 1.0

    def test_naive_datetime_handled(self):
        naive = datetime.now() - timedelta(days=1)
        score = _freshness_score(naive)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests : score_candidate
# ---------------------------------------------------------------------------

class TestScoreCandidate:

    def _user_emb(self):
        return make_emb(1.0)

    def test_output_in_zero_one(self):
        c = make_candidate()
        s = score_candidate(c, self._user_emb())
        assert 0.0 <= s <= 1.0

    def test_toxic_post_scores_lower(self):
        c_clean = make_candidate(toxicity=0.0)
        c_toxic = make_candidate(toxicity=0.9)
        u = self._user_emb()
        assert score_candidate(c_clean, u) > score_candidate(c_toxic, u)

    def test_popular_post_scores_higher(self):
        c_pop    = make_candidate(likes=5000, views=20000)
        c_unpop  = make_candidate(likes=0, views=0)
        u = self._user_emb()
        assert score_candidate(c_pop, u) > score_candidate(c_unpop, u)

    def test_fresh_post_scores_higher(self):
        now  = datetime.now(timezone.utc)
        old  = now - timedelta(days=60)
        u    = self._user_emb()
        c_fresh = make_candidate(created_at=now)
        c_old   = make_candidate(created_at=old)
        assert score_candidate(c_fresh, u) > score_candidate(c_old, u)

    def test_similar_post_scores_higher(self):
        u = make_emb(1.0)
        c_sim  = make_candidate(post_emb=make_emb(1.0))    # identique → sim=1
        c_diff = make_candidate(post_emb=[-x for x in make_emb(1.0)])  # opposé → sim=0
        assert score_candidate(c_sim, u) > score_candidate(c_diff, u)

    def test_missing_embedding_does_not_crash(self):
        c = make_candidate()
        c["embedding"] = None
        s = score_candidate(c, self._user_emb())
        assert 0.0 <= s <= 1.0

    def test_custom_weights_applied(self):
        # Poids 100% popularité
        w = {"similarity": 0.0, "clean": 0.0, "popularity": 1.0, "freshness": 0.0}
        c_pop   = make_candidate(likes=10_000, views=0)
        c_unpop = make_candidate(likes=0, views=0)
        u = self._user_emb()
        assert score_candidate(c_pop, u, weights=w) > score_candidate(c_unpop, u, weights=w)

    def test_returns_float(self):
        c = make_candidate()
        assert isinstance(score_candidate(c, self._user_emb()), float)


# ---------------------------------------------------------------------------
# Tests : rank_candidates
# ---------------------------------------------------------------------------

class TestRankCandidates:

    def _user_emb(self):
        return make_emb(1.0)

    def test_empty_list_returns_empty(self):
        assert rank_candidates([], self._user_emb()) == []

    def test_rank_score_field_added(self):
        c = make_candidate()
        result = rank_candidates([c], self._user_emb())
        assert "rank_score" in result[0]

    def test_sorted_descending(self):
        now = datetime.now(timezone.utc)
        c1 = make_candidate(toxicity=0.9, likes=0, created_at=now, cid="bad")
        c2 = make_candidate(toxicity=0.0, likes=5000, created_at=now, cid="good")
        ranked = rank_candidates([c1, c2], self._user_emb())
        assert ranked[0]["id"] == "good"
        assert ranked[0]["rank_score"] >= ranked[1]["rank_score"]

    def test_no_mutation_of_input(self):
        c = make_candidate()
        original_keys = set(c.keys())
        rank_candidates([c], self._user_emb())
        assert set(c.keys()) == original_keys  # pas de rank_score ajouté sur l'original

    def test_none_user_embedding_does_not_crash(self):
        candidates = [make_candidate(), make_candidate(cid="post_2")]
        result = rank_candidates(candidates, user_embedding=None)
        assert len(result) == 2

    def test_scores_monotonically_decreasing(self):
        now = datetime.now(timezone.utc)
        candidates = [
            make_candidate(toxicity=0.8, likes=0,     cid="c1"),
            make_candidate(toxicity=0.0, likes=1000,  cid="c2"),
            make_candidate(toxicity=0.3, likes=200,   cid="c3"),
        ]
        ranked = rank_candidates(candidates, self._user_emb())
        scores = [r["rank_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests : cohérence globale des poids
# ---------------------------------------------------------------------------

class TestRankerConfig:

    def test_weights_sum_to_one(self):
        total = sum(RANKER_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_all_weights_positive(self):
        for key, w in RANKER_WEIGHTS.items():
            assert w >= 0.0, f"Poids négatif : {key}={w}"

    def test_expected_keys_present(self):
        assert set(RANKER_WEIGHTS.keys()) == {
            "similarity", "clean", "popularity", "freshness"
        }