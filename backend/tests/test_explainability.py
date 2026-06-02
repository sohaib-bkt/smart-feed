"""
Tests unitaires pour app/services/explainability.py

Couvre :
- build_explanation : chaque dimension isolément + cas limites
- annotate_feed     : mutation en place, idempotence, liste vide
- Cohérence globale : summary tronqué à 3 raisons, fallback, types

Lancer :
    cd backend
    pytest tests/test_explainability.py -v
"""

from __future__ import annotations

import pytest
from app.services.explainability import annotate_feed, build_explanation


def _post(**kwargs) -> dict:
    """Construit un post minimal avec des valeurs neutres par défaut."""
    defaults = {
        "id": "p1",
        "score": 0.5,
        "cosine_sim": 0.0,
        "cf_score": 0.5,
        "category": "TECH",
        "recency": 0.5,
        "modal": "text",
        "toxicity_score": 0.0,
    }
    defaults.update(kwargs)
    return defaults


def _prefs(**kwargs) -> dict:
    defaults = {
        "interests": [],
        "content_type": "all",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# build_explanation — structure de base
# ---------------------------------------------------------------------------

class TestBuildExplanationStructure:

    def test_returns_required_keys(self):
        result = build_explanation(_post(), _prefs(), [])
        assert set(result.keys()) == {"reasons", "summary", "detail", "score"}

    def test_reasons_is_list(self):
        result = build_explanation(_post(), _prefs(), [])
        assert isinstance(result["reasons"], list)

    def test_summary_is_string(self):
        result = build_explanation(_post(), _prefs(), [])
        assert isinstance(result["summary"], str)

    def test_detail_is_dict(self):
        result = build_explanation(_post(), _prefs(), [])
        assert isinstance(result["detail"], dict)

    def test_score_is_float(self):
        result = build_explanation(_post(score=0.75), _prefs(), [])
        assert isinstance(result["score"], float)

    def test_detail_contains_all_dimension_keys(self):
        result = build_explanation(_post(), _prefs(), [])
        for key in ("similarity", "collaborative", "category_match",
                    "recency", "modal", "toxicity", "xgb_score"):
            assert key in result["detail"], f"Clé manquante : {key}"


# ---------------------------------------------------------------------------
# Dimension 1 : similarité sémantique
# ---------------------------------------------------------------------------

class TestDimensionSimilarity:

    def test_high_similarity_generates_reason(self):
        result = build_explanation(_post(cosine_sim=0.80), _prefs(), [])
        assert any("similaire" in r for r in result["reasons"])

    def test_medium_similarity_generates_reason(self):
        result = build_explanation(_post(cosine_sim=0.60), _prefs(), [])
        assert any("similaire" in r for r in result["reasons"])

    def test_partial_similarity_generates_reason(self):
        result = build_explanation(_post(cosine_sim=0.40), _prefs(), [])
        assert any("partiellement" in r for r in result["reasons"])

    def test_low_similarity_no_reason(self):
        result = build_explanation(_post(cosine_sim=0.10), _prefs(), [])
        assert not any("similaire" in r or "partiellement" in r
                       for r in result["reasons"])

    def test_similarity_stored_in_detail(self):
        result = build_explanation(_post(cosine_sim=0.65), _prefs(), [])
        assert result["detail"]["similarity"] == pytest.approx(0.65, abs=0.001)

    def test_similarity_fallback_from_score_detail(self):
        """cosine_sim absent → doit lire score_detail.cosine_sim."""
        post = {
            "id": "p1",
            "score": 0.5,
            "category": "TECH",
            "modal": "text",
            "toxicity_score": 0.0,
            "score_detail": {"cosine_sim": 0.80},
        }
        result = build_explanation(post, _prefs(), [])
        assert any("similaire" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# Dimension 2 : filtrage collaboratif
# ---------------------------------------------------------------------------

class TestDimensionCollaborative:

    def test_high_cf_generates_strong_reason(self):
        result = build_explanation(_post(cf_score=0.80), _prefs(), [])
        assert any("très populaire" in r for r in result["reasons"])

    def test_medium_cf_generates_weak_reason(self):
        result = build_explanation(_post(cf_score=0.65), _prefs(), [])
        assert any("apprécié" in r for r in result["reasons"])

    def test_low_cf_no_reason(self):
        result = build_explanation(_post(cf_score=0.30), _prefs(), [])
        assert not any("populaire" in r or "apprécié" in r
                       for r in result["reasons"])

    def test_cf_stored_in_detail(self):
        result = build_explanation(_post(cf_score=0.75), _prefs(), [])
        assert result["detail"]["collaborative"] == pytest.approx(0.75, abs=0.001)

    def test_cf_fallback_from_score_detail(self):
        """cf_score absent → doit lire score_detail.collaborative."""
        post = {
            "id": "p1",
            "score": 0.5,
            "category": "TECH",
            "modal": "text",
            "toxicity_score": 0.0,
            "score_detail": {"collaborative": 0.80},
        }
        result = build_explanation(post, _prefs(), [])
        assert any("très populaire" in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# Dimension 3 : catégorie favorite
# ---------------------------------------------------------------------------

class TestDimensionCategory:

    def test_matching_category_generates_reason(self):
        result = build_explanation(
            _post(category="TECH"),
            _prefs(interests=["TECH", "SCIENCE"]),
            []
        )
        assert any("TECH" in r for r in result["reasons"])

    def test_non_matching_category_no_reason(self):
        result = build_explanation(
            _post(category="SPORTS"),
            _prefs(interests=["TECH"]),
            []
        )
        assert not any("SPORTS" in r for r in result["reasons"])

    def test_empty_interests_no_reason(self):
        result = build_explanation(
            _post(category="TECH"),
            _prefs(interests=[]),
            []
        )
        assert not any("favoris" in r for r in result["reasons"])

    def test_category_match_true_in_detail(self):
        result = build_explanation(
            _post(category="SCIENCE"),
            _prefs(interests=["SCIENCE"]),
            []
        )
        assert result["detail"]["category_match"] is True

    def test_category_match_false_in_detail(self):
        result = build_explanation(
            _post(category="SPORTS"),
            _prefs(interests=["TECH"]),
            []
        )
        assert result["detail"]["category_match"] is False


# ---------------------------------------------------------------------------
# Dimension 4 : récence
# ---------------------------------------------------------------------------

class TestDimensionRecency:

    def test_very_recent_generates_strong_reason(self):
        result = build_explanation(_post(recency=0.90), _prefs(), [])
        assert any("très récent" in r for r in result["reasons"])

    def test_recent_generates_reason(self):
        result = build_explanation(_post(recency=0.70), _prefs(), [])
        assert any("récent" in r for r in result["reasons"])

    def test_old_content_no_reason(self):
        result = build_explanation(_post(recency=0.30), _prefs(), [])
        assert not any("récent" in r for r in result["reasons"])

    def test_recency_stored_in_detail(self):
        result = build_explanation(_post(recency=0.88), _prefs(), [])
        assert result["detail"]["recency"] == pytest.approx(0.88, abs=0.01)


# ---------------------------------------------------------------------------
# Dimension 5 : modalité
# ---------------------------------------------------------------------------

class TestDimensionModal:

    def test_matching_modal_generates_reason(self):
        result = build_explanation(
            _post(modal="video"),
            _prefs(content_type="video"),
            []
        )
        assert any("vidéo courte" in r for r in result["reasons"])

    def test_content_type_all_no_modal_reason(self):
        result = build_explanation(
            _post(modal="video"),
            _prefs(content_type="all"),
            []
        )
        assert not any("vidéo" in r for r in result["reasons"])

    def test_non_matching_modal_no_reason(self):
        result = build_explanation(
            _post(modal="text"),
            _prefs(content_type="video"),
            []
        )
        assert not any("format" in r for r in result["reasons"])

    def test_modal_stored_in_detail(self):
        result = build_explanation(_post(modal="image"), _prefs(), [])
        assert result["detail"]["modal"] == "image"

    @pytest.mark.parametrize("modal,label", [
        ("text",  "article texte"),
        ("image", "post image"),
        ("video", "vidéo courte"),
    ])
    def test_modal_labels(self, modal: str, label: str):
        result = build_explanation(
            _post(modal=modal),
            _prefs(content_type=modal),
            []
        )
        assert any(label in r for r in result["reasons"])


# ---------------------------------------------------------------------------
# Dimension 6 : toxicité / santé
# ---------------------------------------------------------------------------

class TestDimensionToxicity:

    def test_very_clean_content_generates_reason(self):
        result = build_explanation(_post(toxicity_score=0.01), _prefs(), [])
        assert any("très sain" in r for r in result["reasons"])

    def test_low_toxicity_generates_reason(self):
        result = build_explanation(_post(toxicity_score=0.05), _prefs(), [])
        assert any("faible toxicité" in r for r in result["reasons"])

    def test_high_toxicity_no_positive_reason(self):
        result = build_explanation(_post(toxicity_score=0.50), _prefs(), [])
        assert not any("sain" in r or "toxicité" in r
                       for r in result["reasons"])

    def test_toxicity_stored_in_detail(self):
        result = build_explanation(_post(toxicity_score=0.07), _prefs(), [])
        assert result["detail"]["toxicity"] == pytest.approx(0.07, abs=0.001)


# ---------------------------------------------------------------------------
# Dimension 7 : score XGBoost
# ---------------------------------------------------------------------------

class TestDimensionXgbScore:

    def test_xgb_score_from_xgb_score_field(self):
        result = build_explanation(_post(xgb_score=0.8765), _prefs(), [])
        assert result["score"] == pytest.approx(0.8765, abs=0.0001)
        assert result["detail"]["xgb_score"] == pytest.approx(0.8765, abs=0.0001)

    def test_xgb_score_fallback_to_rank_score(self):
        post = _post()
        post.pop("score", None)
        post["rank_score"] = 0.7654
        result = build_explanation(post, _prefs(), [])
        assert result["score"] == pytest.approx(0.7654, abs=0.0001)

    def test_xgb_score_fallback_to_score(self):
        result = build_explanation(_post(score=0.6543), _prefs(), [])
        assert result["score"] == pytest.approx(0.6543, abs=0.0001)

    def test_xgb_score_zero_when_absent(self):
        post = {"id": "p1", "category": "TECH", "modal": "text"}
        result = build_explanation(post, _prefs(), [])
        assert result["score"] == pytest.approx(0.0, abs=0.0001)


# ---------------------------------------------------------------------------
# Fallback & robustesse
# ---------------------------------------------------------------------------

class TestFallbackAndRobustness:

    def test_fallback_reason_when_no_dimension_triggered(self):
        """Un post neutre doit toujours avoir au moins une raison."""
        post = _post(
            cosine_sim=0.0,
            cf_score=0.5,
            category="SPORTS",
            recency=0.5,
            modal="text",
            toxicity_score=0.5,
        )
        result = build_explanation(post, _prefs(interests=[]), [])
        assert len(result["reasons"]) >= 1
        assert "profil" in result["reasons"][0]

    def test_summary_contains_prefix(self):
        result = build_explanation(_post(), _prefs(), [])
        assert result["summary"].startswith("Recommandé car")

    def test_summary_uses_at_most_three_reasons(self):
        """Le résumé ne doit pas dépasser 3 raisons."""
        post = _post(
            cosine_sim=0.80,
            cf_score=0.80,
            category="TECH",
            recency=0.90,
            modal="video",
            toxicity_score=0.01,
        )
        prefs = _prefs(interests=["TECH"], content_type="video")
        result = build_explanation(post, prefs, [])
        # Il peut y avoir plus de 3 raisons globalement
        assert len(result["reasons"]) >= 3
        # Le résumé doit se limiter à 3
        summary_reasons = result["summary"].replace("Recommandé car : ", "")
        parts = summary_reasons.split(", ")
        assert len(parts) <= 3

    def test_missing_fields_do_not_crash(self):
        """Un post incomplet ne doit pas lever d'exception."""
        result = build_explanation({}, _prefs(), [])
        assert "reasons" in result

    def test_none_values_do_not_crash(self):
        post = {
            "cosine_sim": None,
            "cf_score": None,
            "category": None,
            "recency": None,
            "modal": None,
            "toxicity_score": None,
            "score": None,
        }
        # Ne doit pas lever d'exception
        try:
            result = build_explanation(post, _prefs(), [])
            assert "reasons" in result
        except (TypeError, ValueError) as e:
            pytest.fail(f"build_explanation a levé une exception inattendue : {e}")

    def test_interactions_parameter_accepted(self):
        """Le paramètre interactions doit être accepté sans erreur."""
        interactions = [{"post_id": "x", "action": "like"}]
        result = build_explanation(_post(), _prefs(), interactions)
        assert "reasons" in result


# ---------------------------------------------------------------------------
# annotate_feed
# ---------------------------------------------------------------------------

class TestAnnotateFeed:

    def test_annotate_adds_explanation_field(self):
        feed = [_post(id="p1"), _post(id="p2")]
        annotate_feed(feed, _prefs(), [])
        for post in feed:
            assert "explanation" in post

    def test_annotate_modifies_in_place(self):
        feed = [_post(id="p1")]
        original_ref = feed[0]
        annotate_feed(feed, _prefs(), [])
        assert feed[0] is original_ref
        assert "explanation" in feed[0]

    def test_annotate_empty_feed_does_not_crash(self):
        annotate_feed([], _prefs(), [])  # ne doit pas lever d'exception

    def test_annotate_explanation_structure(self):
        feed = [_post(id="p1")]
        annotate_feed(feed, _prefs(), [])
        exp = feed[0]["explanation"]
        assert set(exp.keys()) == {"reasons", "summary", "detail", "score"}

    def test_annotate_all_posts(self):
        n = 5
        feed = [_post(id=f"p{i}") for i in range(n)]
        annotate_feed(feed, _prefs(), [])
        assert all("explanation" in p for p in feed)

    def test_annotate_with_matching_interests(self):
        feed = [_post(category="TECH", cosine_sim=0.80)]
        prefs = _prefs(interests=["TECH"])
        annotate_feed(feed, prefs, [])
        reasons = feed[0]["explanation"]["reasons"]
        assert any("TECH" in r for r in reasons)

    def test_annotate_idempotent(self):
        """Appeler annotate_feed deux fois ne doit pas dupliquer les raisons."""
        feed = [_post(id="p1")]
        prefs = _prefs()
        annotate_feed(feed, prefs, [])
        first_reasons = list(feed[0]["explanation"]["reasons"])
        annotate_feed(feed, prefs, [])
        second_reasons = list(feed[0]["explanation"]["reasons"])
        assert first_reasons == second_reasons


# ---------------------------------------------------------------------------
# Cohérence multi-dimensions
# ---------------------------------------------------------------------------

class TestMultiDimensionCombinations:

    def test_ideal_post_has_many_reasons(self):
        """Un post parfait doit déclencher plusieurs dimensions."""
        post = _post(
            cosine_sim=0.85,
            cf_score=0.80,
            category="SCIENCE",
            recency=0.90,
            modal="text",
            toxicity_score=0.01,
        )
        prefs = _prefs(interests=["SCIENCE"], content_type="all")
        result = build_explanation(post, prefs, [])
        assert len(result["reasons"]) >= 3

    def test_poor_post_uses_fallback(self):
        """Un post avec tous les scores faibles doit utiliser le fallback."""
        post = _post(
            cosine_sim=0.10,
            cf_score=0.30,
            category="SPORTS",
            recency=0.10,
            modal="text",
            toxicity_score=0.50,
        )
        prefs = _prefs(interests=["TECH"])
        result = build_explanation(post, prefs, [])
        assert "profil" in result["reasons"][0]

    def test_detail_keys_always_present_regardless_of_reasons(self):
        """Les clés de détail doivent toujours être présentes."""
        post = _post()
        result = build_explanation(post, _prefs(), [])
        for key in ("similarity", "collaborative", "category_match",
                    "recency", "modal", "toxicity", "xgb_score"):
            assert key in result["detail"]


if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])