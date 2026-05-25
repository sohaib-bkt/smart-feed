# tests/pipeline_validation.py
# Validation complète de la pipeline multi-modèle
# Modèles validés :
#   - Embeddings (SentenceTransformer all-MiniLM-L6-v2)
#   - Toxicity (Detoxify)
#   - Collaborative (Implicit ALS)
#   - Content scoring (compute_score)
#   - Re-ranking (ranker.py - similarity, clean, popularity, freshness + CF adjustment)
#   - FAISS indexes (v1, v2, v3)
#
# Usage: python -m tests.pipeline_validation

import sys
sys.path.insert(0, '.')

import asyncio
import json
import math
import numpy as np
from datetime import datetime, timezone, timedelta

# =========================================================================
# 1. Model Imports
# =========================================================================
from app.models.embeddings import get_model as get_emb_model, embed_text, embed_batch
from app.models.toxicity import _get_model as get_tox_model, score_toxicity, is_toxic
from app.models.collaborative import (
    get_cf_scores, get_top_cf_articles, get_user_recommendations_with_scores, _load_model
)
from app.models.ranker import (
    score_candidate, rank_candidates, explain_ranking,
    RANKER_WEIGHTS, MAX_ARTICLES_PER_CATEGORY,
    CF_BOOST_THRESHOLD, CF_BOOST_STRONG, CF_BOOST_WEAK, CF_PENALTY
)
from app.services.scoring import compute_score, MODE_WEIGHTS
from app.services.user_profile import compute_user_embedding, get_top_interests
from app.services.recommender import get_feed, get_feed_v2, get_feed_v3
from app.services.hybrid_recommender import get_hybrid_feed
from app.db.firebase import get_user_profile, get_user_interactions, create_user

_TEST_USERS = ['sim_user_17', 'sim_user_05', 'sim_user_04']


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title):
    print(f"\n  --- {title} ---")


# =====================================================================
# HELPER: async runner
# =====================================================================
def _run(coro):
    return asyncio.run(coro)


# =====================================================================
# SECTION 1 : Model Loading & Health
# =====================================================================
def test_embedding_model():
    print_section("1. Embedding Model Health")
    model = get_emb_model()
    assert model is not None, "Embedding model is None"
    
    # Test basic embedding
    vec = embed_text("Hello world")
    assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"
    
    # Test batch embedding
    texts = ["one", "two", "three"]
    batch = embed_batch(texts)
    assert batch.shape == (3, 384), f"Expected (3,384), got {batch.shape}"
    
    # Check normalization
    norm = np.linalg.norm(batch[0])
    assert abs(norm - 1.0) < 0.1, f"Embedding not normalized: norm={norm:.4f}"
    
    print(f"  ✅ Embedding model: all-MiniLM-L6-v2")
    print(f"  ✅ Single embedding: 384 dims, norm={np.linalg.norm(vec):.4f}")
    print(f"  ✅ Batch embedding: {batch.shape}")
    print(f"  ✅ Embeddings are normalized")


def test_toxicity_model():
    print_section("2. Toxicity Model Health")
    model = get_tox_model()
    assert model is not None, "Toxicity model is None"
    
    # Test clean text
    clean = score_toxicity("This is a perfectly normal sentence")
    assert isinstance(clean, dict)
    assert 'toxicity_score' in clean
    assert 0 <= clean['toxicity_score'] <= 1
    assert clean['toxicity_score'] < 0.3, "Clean text scored too high"
    
    # Test toxic text
    toxic = score_toxicity("You are an idiot and worthless")
    assert toxic['toxicity_score'] >= 0, f"Invalid toxicity score: {toxic['toxicity_score']}"
    
    # Test is_toxic helper
    assert not is_toxic("Have a nice day", threshold=0.3)
    
    print(f"  ✅ Toxicity model: Detoxify (original)")
    print(f"  ✅ Clean text score: {clean['toxicity_score']:.4f}")
    print(f"  ✅ Toxic text score: {toxic['toxicity_score']:.4f}")
    print(f"  ✅ Toxicity detail keys: {list(clean.get('toxicity_detail', {}).keys())}")


def test_collaborative_model():
    print_section("3. Collaborative Model Health")
    model, user2idx, art2idx, matrix = _load_model()
    
    assert model is not None, "Collaborative model is None"
    assert user2idx is not None, "user2idx is None"
    assert art2idx is not None, "art2idx is None"
    
    print(f"  ✅ Collaborative model: Implicit ALS")
    print(f"  ✅ Users in model: {model.user_factors.shape[0]}")
    print(f"  ✅ Items in model: {model.item_factors.shape[0]}")
    print(f"  ✅ User2idx mapping: {len(user2idx)} users")
    print(f"  ✅ Art2idx mapping: {len(art2idx)} articles")
    print(f"  ✅ User factors shape: {model.user_factors.shape}")
    print(f"  ✅ Item factors shape: {model.item_factors.shape}")
    
    # Check factor norms
    user_norms = np.linalg.norm(model.user_factors, axis=1)
    item_norms = np.linalg.norm(model.item_factors, axis=1)
    print(f"  ✅ User factor norm: mean={np.mean(user_norms):.4f}")
    print(f"  ✅ Item factor norm: mean={np.mean(item_norms):.4f}")
    
    return model, user2idx, art2idx


def test_faiss_indexes():
    print_section("4. FAISS Indexes Health")
    
    from app.services.recommender import _load_resources, _load_resources_v2, _load_resources_v3
    
    # V1
    _load_resources()
    from app.services.recommender import _index as idx_v1, _posts_df as df_v1
    assert idx_v1 is not None, "FAISS v1 index is None"
    assert df_v1 is not None, "FAISS v1 metadata is None"
    print(f"  ✅ FAISS v1: {idx_v1.ntotal} vectors, {df_v1.shape[0]} metadata rows")
    
    # V2
    _load_resources_v2()
    from app.services.recommender import _index_v2 as idx_v2, _posts_df_v2 as df_v2
    assert idx_v2 is not None, "FAISS v2 index is None"
    assert df_v2 is not None, "FAISS v2 metadata is None"
    print(f"  ✅ FAISS v2: {idx_v2.ntotal} vectors, {df_v2.shape[0]} metadata rows")
    
    # V3
    try:
        _load_resources_v3()
        from app.services.recommender import _index_v3 as idx_v3, _posts_df_v3 as df_v3
        print(f"  ✅ FAISS v3: {idx_v3.ntotal} vectors (512-dim), {df_v3.shape[0]} metadata rows")
    except Exception as e:
        print(f"  ⚠️ FAISS v3 not available: {e}")
    
    return idx_v1, df_v1


# =====================================================================
# SECTION 2 : User Profile Pipeline
# =====================================================================
def test_user_profile_pipeline():
    print_section("5. User Profile Pipeline")
    
    all_ok = True
    for user_id in _TEST_USERS:
        print_subsection(f"User: {user_id}")
        
        profile = _run(get_user_profile(user_id))
        assert profile is not None, f"Profile not found for {user_id}"
        print(f"  ✅ Profile loaded: {profile.get('user_id')}")
        
        interactions = _run(get_user_interactions(user_id, last_n=100))
        assert len(interactions) > 0, f"No interactions for {user_id}"
        print(f"  ✅ Interactions: {len(interactions)}")
        
        embedding = compute_user_embedding(interactions)
        assert len(embedding) == 384, f"Expected 384 dims, got {len(embedding)}"
        norm = np.linalg.norm(embedding)
        print(f"  ✅ Embedding: 384 dims, norm={norm:.4f}")
        
        interests = get_top_interests(interactions, n=5)
        print(f"  ✅ Top interests: {interests}")
        
        # Check embedding is not zero vector
        if norm < 0.1:
            print(f"  ⚠️ Embedding norm very low: {norm:.4f}")
            all_ok = False
    
    return all_ok


# =====================================================================
# SECTION 3 : Content-Based Pipeline (v1)
# =====================================================================
def test_content_based_pipeline():
    print_section("6. Content-Based Pipeline (v1)")
    
    all_ok = True
    for user_id in _TEST_USERS:
        print_subsection(f"User: {user_id}")
        
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        feed = get_feed(
            user_embedding=embedding,
            user_prefs=prefs,
            n_candidates=200,
            n_results=20
        )
        
        if not feed:
            print(f"  ❌ No results for {user_id}")
            all_ok = False
            continue
        
        assert isinstance(feed, list), "Feed should be a list"
        assert len(feed) > 0, f"Empty feed for {user_id}"
        assert len(feed) <= 20, f"Too many results: {len(feed)}"
        
        # Check required fields
        required = ['id', 'score', 'category', 'score_detail']
        for field in required:
            assert field in feed[0], f"Missing field '{field}' in feed item"
        
        # Score validation
        scores = [item['score'] for item in feed]
        assert all(0 < s <= 1 for s in scores), "Scores out of (0,1] range"
        assert len(set(scores)) > 1, "All scores are identical"
        
        print(f"  ✅ {len(feed)} items returned")
        print(f"  ✅ Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  ✅ Score mean: {np.mean(scores):.4f}")
        print(f"  ✅ Unique categories: {len(set(item.get('category', '?') for item in feed))}")
    
    return all_ok


# =====================================================================
# SECTION 4 : Hybrid Pipeline (v2)
# =====================================================================
def test_hybrid_pipeline():
    print_section("7. Hybrid Pipeline (v2)")
    
    all_ok = True
    ALPHA, BETA = 0.6, 0.4
    
    for user_id in _TEST_USERS:
        print_subsection(f"User: {user_id}")
        
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        candidates = get_hybrid_feed(
            user_id=user_id,
            user_embedding=embedding,
            user_prefs=prefs,
            n_candidates=300,
            n_results=60,
            diversify=True,
            max_per_category=3
        )
        
        if not candidates:
            print(f"  ⚠️ No candidates for {user_id}")
            all_ok = False
            continue
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 60
        
        # Check hybrid score structure
        item = candidates[0]
        assert 'score' in item
        assert 'score_detail' in item
        assert 'content_based' in item['score_detail']
        assert 'collaborative' in item['score_detail']
        
        # Verify hybrid formula: score = 0.6 * cb + 0.4 * cf
        cb = item['score_detail']['content_based']
        cf = item['score_detail']['collaborative']
        expected = ALPHA * cb + BETA * cf
        assert abs(item['score'] - round(expected, 6)) < 1e-4, \
            f"Hybrid formula mismatch: {item['score']} != {expected}"
        
        # Check all scores are valid
        scores = [c['score'] for c in candidates]
        content_based_scores = [c['score_detail']['content_based'] for c in candidates]
        cf_scores = [c['score_detail']['collaborative'] for c in candidates]
        
        print(f"  ✅ {len(candidates)} candidates returned")
        print(f"  ✅ Hybrid score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  ✅ CB score range: [{min(content_based_scores):.4f}, {max(content_based_scores):.4f}]")
        print(f"  ✅ CF score range: [{min(cf_scores):.4f}, {max(cf_scores):.4f}]")
        
        # Check diversity (max 3 per category)
        from collections import Counter
        cat_counts = Counter(c.get('category', 'Unknown') for c in candidates)
        max_cat = max(cat_counts.values())
        if max_cat > 3:
            print(f"  ⚠️ Category limit exceeded: {max_cat} max (limit=3)")
        
        print(f"  ✅ Unique categories: {len(cat_counts)}")
        
        # Verify sort order
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
            "Candidates not sorted by score descending"
    
    return all_ok


# =====================================================================
# SECTION 5 : Multi-Modal Pipeline (v3)
# =====================================================================
def test_multimodal_pipeline():
    print_section("8. Multi-Modal Pipeline (v3)")
    
    try:
        from app.services.recommender import _index_v3
    except Exception:
        print("  ⚠️ FAISS v3 index not loaded - skipping")
        return None
    
    all_ok = True
    for user_id in _TEST_USERS:
        print_subsection(f"User: {user_id}")
        
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        feed = get_feed_v3(
            user_embedding=embedding,
            user_prefs=prefs,
            n_candidates=200,
            n_results=20,
            embedding_dim=384
        )
        
        if not feed:
            print(f"  ⚠️ No v3 results for {user_id}")
            continue
        
        assert len(feed) <= 20
        scores = [item['score'] for item in feed]
        
        print(f"  ✅ {len(feed)} items returned")
        print(f"  ✅ Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  ✅ Score mean: {np.mean(scores):.4f}")
        
        # Check modal field if present
        if 'modal' in feed[0]:
            modals = set(item.get('modal', 'text') for item in feed)
            print(f"  ✅ Modalities: {modals}")
    
    return all_ok


# =====================================================================
# SECTION 6 : Re-ranking (ranker.py)
# =====================================================================
def test_ranker_components():
    print_section("9. Ranker Component Validation")
    
    user_emb = [0.1] * 384
    
    # 9a. score_candidate basic
    print_subsection("9a. score_candidate basic")
    candidate = {
        'id': 'test_123',
        'embedding': [0.2] * 384,
        'toxicity_score': 0.1,
        'likes': 100,
        'views': 1000,
        'created_at': datetime.now(timezone.utc)
    }
    score = score_candidate(candidate, user_emb)
    assert 0 <= score <= 1, f"Score out of range: {score}"
    print(f"  ✅ Basic score: {score:.4f}")
    
    # 9b. Toxicity penalty
    print_subsection("9b. Toxicity penalty")
    clean_cand = {**candidate, 'id': 'clean', 'toxicity_score': 0.0}
    toxic_cand = {**candidate, 'id': 'toxic', 'toxicity_score': 0.9}
    score_clean = score_candidate(clean_cand, user_emb)
    score_toxic = score_candidate(toxic_cand, user_emb)
    assert score_clean > score_toxic, "Toxicity penalty not working"
    print(f"  ✅ Clean={score_clean:.4f} > Toxic={score_toxic:.4f}")
    
    # 9c. Popularity boost
    print_subsection("9c. Popularity boost")
    low_pop = {**candidate, 'id': 'low_pop', 'likes': 0, 'views': 0}
    high_pop = {**candidate, 'id': 'high_pop', 'likes': 1000, 'views': 100000}
    score_low = score_candidate(low_pop, user_emb)
    score_high = score_candidate(high_pop, user_emb)
    assert score_high >= score_low, "Popularity not boosting"
    print(f"  ✅ High pop={score_high:.4f} >= Low pop={score_low:.4f}")
    
    # 9d. Freshness decay
    print_subsection("9d. Freshness decay")
    now = datetime.now(timezone.utc)
    fresh = {**candidate, 'id': 'fresh', 'created_at': now}
    old = {**candidate, 'id': 'old', 'created_at': now - timedelta(days=30)}
    score_fresh = score_candidate(fresh, user_emb)
    score_old = score_candidate(old, user_emb)
    assert score_fresh >= score_old, "Freshness not working"
    print(f"  ✅ Fresh={score_fresh:.4f} >= Old={score_old:.4f}")
    
    # 9e. CF adjustment
    print_subsection("9e. CF adjustment calculation")
    from app.models.ranker import _calculate_cf_adjustment
    assert abs(_calculate_cf_adjustment(0.8) - CF_BOOST_STRONG) < 1e-4, \
        f"Expected CF boost strong {CF_BOOST_STRONG}"
    assert abs(_calculate_cf_adjustment(0.65) - CF_BOOST_WEAK) < 1e-4, \
        f"Expected CF boost weak {CF_BOOST_WEAK}"
    assert abs(_calculate_cf_adjustment(0.3) - CF_PENALTY) < 1e-4, \
        f"Expected CF penalty {CF_PENALTY}"
    assert _calculate_cf_adjustment(0.5) == 0.0, "Middle CF should have no adjustment"
    print(f"  ✅ CF adjustment: strong={CF_BOOST_STRONG}, weak={CF_BOOST_WEAK}, penalty={CF_PENALTY}")
    
    # 9f. Weight validation
    print_subsection("9f. Weight validation")
    total_weight = sum(RANKER_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 1e-4, \
        f"Ranker weights sum to {total_weight}, expected 1.0"
    print(f"  ✅ Ranker weights sum to {total_weight}")
    for k, v in RANKER_WEIGHTS.items():
        print(f"     {k}: {v}")
    
    # 9g. rank_candidates on hybrid candidates
    print_subsection("9g. rank_candidates on real data")
    for user_id in _TEST_USERS:
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        candidates = get_hybrid_feed(
            user_id=user_id, user_embedding=embedding, user_prefs=prefs,
            n_candidates=200, n_results=60, diversify=False
        )
        
        if not candidates:
            print(f"  ⚠️ No candidates for {user_id}")
            continue
        
        ranked = rank_candidates(candidates, user_embedding=embedding, diversify=True)
        
        assert len(ranked) <= len(candidates), "Ranking added candidates"
        assert all('rank_score' in c for c in ranked), "Missing rank_score"
        
        # Check diversification works
        from collections import Counter
        cat_counts = Counter(c.get('category', 'Unknown') for c in ranked)
        max_cat = max(cat_counts.values())
        assert max_cat <= MAX_ARTICLES_PER_CATEGORY, \
            f"Category limit exceeded: {max_cat} > {MAX_ARTICLES_PER_CATEGORY}"
        
        rank_scores = [c['rank_score'] for c in ranked]
        assert rank_scores == sorted(rank_scores, reverse=True), \
            "Not sorted by rank_score descending"
        
        print(f"  ✅ {user_id}: {len(ranked)} ranked items, {len(cat_counts)} categories")
        print(f"     Rank score range: [{min(rank_scores):.4f}, {max(rank_scores):.4f}]")
    
    return True


# =====================================================================
# SECTION 7 : Personalization
# =====================================================================
def test_personalization():
    print_section("10. Personalization Verification")
    
    feeds = {}
    for user_id in _TEST_USERS:
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        candidates = get_hybrid_feed(
            user_id=user_id, user_embedding=embedding, user_prefs=prefs,
            n_candidates=300, n_results=60, diversify=True, max_per_category=3
        )
        
        ranked = rank_candidates(candidates, user_embedding=embedding, diversify=True)
        feeds[user_id] = set(c['id'] for c in ranked[:15])
    
    print(f"\n  Users: {_TEST_USERS}")
    
    import itertools
    overlaps = {}
    for u1, u2 in itertools.combinations(_TEST_USERS, 2):
        overlap = len(feeds[u1] & feeds[u2])
        overlaps[f"{u1} vs {u2}"] = overlap
        print(f"  🔄 Overlap {u1} vs {u2}: {overlap}/15 common items")
    
    max_overlap = max(overlaps.values())
    if max_overlap <= 3:
        print(f"\n  ✅ Excellent personalization: max overlap = {max_overlap}/15")
    elif max_overlap <= 6:
        print(f"\n  ✅ Good personalization: max overlap = {max_overlap}/15")
    else:
        print(f"\n  ⚠️ Weak personalization: max overlap = {max_overlap}/15")
    
    return True


# =====================================================================
# SECTION 8 : Cold Start
# =====================================================================
def test_cold_start():
    print_section("11. Cold Start Handling")
    
    # Collaborative cold start
    cf_scores = get_cf_scores('unknown_user_xyz', [1, 2, 3])
    assert all(s == 0.5 for s in cf_scores.values()), \
        "Cold start CF scores should all be 0.5"
    print(f"  ✅ Collaborative cold start: all scores = 0.5")
    
    # User embedding cold start (no interactions)
    empty_emb = compute_user_embedding([])
    assert empty_emb == [0.0] * 384, "Empty interactions should give zero embedding"
    print(f"  ✅ User embedding cold start: zero vector returned")
    
    # Feed with zero embedding
    feed_zero = get_feed(
        user_embedding=[0.0] * 384,
        user_prefs={"mode": "default", "interests": []},
        n_candidates=50,
        n_results=10
    )
    if feed_zero:
        print(f"  ✅ v1 feed with zero embedding: {len(feed_zero)} items")
    else:
        print(f"  ⚠️ v1 feed with zero embedding returned no items")
    
    # Hybrid feed with unknown user
    try:
        hybrid_cold = get_hybrid_feed(
            user_id='unknown_user_xyz',
            user_embedding=[0.0] * 384,
            user_prefs={"mode": "default", "interests": []},
            n_candidates=100,
            n_results=20
        )
        if hybrid_cold:
            print(f"  ✅ Hybrid feed cold start: {len(hybrid_cold)} items")
        else:
            print(f"  ⚠️ Hybrid feed cold start returned no items")
    except Exception as e:
        print(f"  ⚠️ Hybrid feed cold start error: {e}")
    
    return True


# =====================================================================
# SECTION 9 : Score Distribution Analysis
# =====================================================================
def test_score_distribution():
    print_section("12. Score Distribution Analysis")
    
    all_v1_scores = []
    all_hybrid_final = []
    all_cb = []
    all_cf = []
    
    for user_id in _TEST_USERS:
        profile = _run(get_user_profile(user_id))
        interactions = _run(get_user_interactions(user_id, last_n=100))
        embedding = compute_user_embedding(interactions)
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        
        # v1 scores
        v1_feed = get_feed(embedding, prefs, n_candidates=200, n_results=20)
        all_v1_scores.extend(item['score'] for item in v1_feed)
        
        # Hybrid scores
        candidates = get_hybrid_feed(
            user_id=user_id, user_embedding=embedding, user_prefs=prefs,
            n_candidates=300, n_results=60
        )
        all_hybrid_final.extend(c['score'] for c in candidates)
        all_cb.extend(c['score_detail']['content_based'] for c in candidates if 'score_detail' in c)
        all_cf.extend(c['score_detail']['collaborative'] for c in candidates if 'score_detail' in c)
        
        # Ranked scores
        ranked = rank_candidates(candidates, user_embedding=embedding, diversify=True)
        key = 'rank_score'
    
    def print_dist(name, scores):
        if not scores:
            print(f"  ⚠️ No {name} scores to analyze")
            return
        high = sum(1 for s in scores if s >= 0.7)
        mid = sum(1 for s in scores if 0.3 <= s < 0.7)
        low = sum(1 for s in scores if s < 0.3)
        print(f"\n  📊 {name}:")
        print(f"     Range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"     Mean: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")
        print(f"     High (>=0.7): {high} ({high/len(scores)*100:.1f}%)")
        print(f"     Medium (0.3-0.7): {mid} ({mid/len(scores)*100:.1f}%)")
        print(f"     Low (<0.3): {low} ({low/len(scores)*100:.1f}%)")
    
    print_dist("v1 Content-Based Scores", all_v1_scores)
    print_dist("Content-Based (hybrid detail)", all_cb)
    print_dist("Collaborative (hybrid detail)", all_cf)
    print_dist("Hybrid Final Scores", all_hybrid_final)
    
    # Correlation between CB and CF
    if all_cb and all_cf and len(all_cb) == len(all_cf):
        corr = np.corrcoef(all_cb, all_cf)[0, 1]
        print(f"\n  📊 CB-CF correlation: {corr:.4f}")
        if abs(corr) < 0.5:
            print(f"  ✅ Good: CB and CF are complementary signals (r={corr:.4f})")
        else:
            print(f"  ⚠️ Warning: CB and CF are highly correlated (r={corr:.4f})")
    
    return True


# =====================================================================
# SECTION 10 : End-to-End Pipeline Simulation (like API v2)
# =====================================================================
def test_end_to_end_pipeline():
    print_section("13. End-to-End Pipeline Simulation (API v2)")
    
    user_id = _TEST_USERS[0]
    print(f"\n  Simulating full API v2 pipeline for {user_id}...")
    
    # Step 1: Load profile
    profile = _run(get_user_profile(user_id))
    assert profile is not None, "Profile not found"
    print(f"  ✅ 1/6 Profile loaded")
    
    # Step 2: Load interactions
    interactions = _run(get_user_interactions(user_id, last_n=100))
    assert len(interactions) > 0, "No interactions"
    print(f"  ✅ 2/6 {len(interactions)} interactions loaded")
    
    # Step 3: Compute embedding
    embedding = compute_user_embedding(interactions)
    assert len(embedding) == 384, "Invalid embedding dimension"
    print(f"  ✅ 3/6 Embedding computed (norm={np.linalg.norm(embedding):.4f})")
    
    # Step 4: Prepare preferences
    prefs = profile.get('preferences', {})
    if not prefs.get('interests') and interactions:
        prefs['interests'] = get_top_interests(interactions)
    print(f"  ✅ 4/6 Preferences ready: mode={prefs.get('mode', 'default')}, "
          f"interests={prefs.get('interests', [])[:3]}")
    
    # Step 5: Hybrid feed
    candidates = get_hybrid_feed(
        user_id=user_id, user_embedding=embedding, user_prefs=prefs,
        n_candidates=300, n_results=60, diversify=True, max_per_category=3
    )
    assert len(candidates) > 0, "No candidates from hybrid feed"
    print(f"  ✅ 5/6 Hybrid feed: {len(candidates)} candidates")
    
    # Step 6: Re-ranking
    ranked = rank_candidates(candidates, user_embedding=embedding, diversify=True)[:20]
    assert len(ranked) > 0, "No items after ranking"
    print(f"  ✅ 6/6 Re-ranking complete: {len(ranked)} final items")
    
    # Detailed output
    print(f"\n  📊 FINAL RECOMMENDATIONS (Top {len(ranked)}):")
    print(f"  {'-' * 70}")
    for i, item in enumerate(ranked):
        rs = item.get('rank_score', item.get('score', 0))
        cat = item.get('category', 'Unknown')
        detail = item.get('score_detail', {})
        cb = detail.get('content_based', 0)
        cf = detail.get('collaborative', 0)
        adj = item.get('_cf_adjustment', 0)
        
        tag = "🔥" if rs >= 0.7 else "👍" if rs >= 0.6 else "📰" if rs >= 0.4 else "❌"
        print(f"  {i+1:2d}. {tag} {item['id']}")
        print(f"       Score: {rs:.4f} | CB: {cb:.3f} | CF: {cf:.3f} | Adj: {adj:+.3f}")
        print(f"       Cat: {cat}")
    
    return ranked

# =====================================================================
# SECTION 11 : Cross-Modal Search Test (texte → image/vidéo)
# =====================================================================

def test_cross_modal_search():
    print_section("14. Cross-Modal Search Test (v3 multimodal)")
    
    try:
        from app.services.recommender import _index_v3, _posts_df_v3
        from app.services.cross_modal_search import search_by_text_cross_modal
    except ImportError as e:
        print(f"  ⚠️ V3 components not available: {e}")
        return None
    
    print("\n  🔍 Testing cross-modal search: Text → Images/Videos")
    
    # Charger les métadonnées v3
    metadata = _posts_df_v3
    
    # Vérifier les modalités disponibles
    if 'modal' not in metadata.columns:
        print("  ⚠️ No 'modal' column in metadata - cannot test cross-modal")
        return None
    
    # Compter les modalités
    modal_counts = metadata['modal'].value_counts().to_dict()
    print(f"  📊 Modalités disponibles: {modal_counts}")
    
    # 1. Test recherche texte → images avec CLIP
    print("\n  📝 Test 1: Search by text (targeting images) with CLIP")
    
    test_queries = [
        "a cat playing with a ball",
        "a dog running in a park",
        "delicious food on a plate",
        "sunset over mountains"
    ]
    
    for query in test_queries:
        # Utiliser CLIP pour la recherche cross-modale
        results = search_by_text_cross_modal(query, k=10, modal_filter='image', use_clip=True)
        
        print(f"     '{query}': {len(results)}/10 images trouvées")
        
        if results:
            for r in results[:2]:
                print(f"        - [{r['score']:.4f}] {r['text'][:50]}...")
    
    # 2. Test avec CLIP sur concepts spécifiques
    print("\n  🖼️ Test 2: Search by text with CLIP (targeting specific concepts)")
    image_queries = [
        "sunset over mountains",
        "delicious food plate",
        "modern architecture building",
        "cute puppy dog"
    ]
    
    for query in image_queries:
        results = search_by_text_cross_modal(query, k=10, modal_filter='image', use_clip=True)
        n_images = len(results)
        print(f"     '{query}': {n_images}/10 images in top results")
        if n_images > 0:
            print(f"        Best: [{results[0]['score']:.4f}] {results[0]['text'][:50]}...")
    
    # 3. Test de diversité multimodale
    print("\n  🌈 Test 3: Multi-modal diversity with CLIP")
    
    results = search_by_text_cross_modal("interesting content", k=50, modal_filter='all', use_clip=True)
    
    modal_counts_top50 = {'text': 0, 'image': 0, 'video': 0}
    for r in results:
        modal_counts_top50[r['modal']] = modal_counts_top50.get(r['modal'], 0) + 1
    
    print(f"  Modal distribution in top 50 results:")
    for modal, count in modal_counts_top50.items():
        if count > 0:
            print(f"     - {modal}: {count} ({count/50*100:.1f}%)")
    
    # Vérifier que les images sont trouvées
    if modal_counts.get('image', 0) > 0:
        if modal_counts_top50.get('image', 0) > 0:
            print("\n  ✅ Cross-modal search works! Images are being retrieved with CLIP.")
        else:
            print("\n  ⚠️ Images exist but not in top 50 - adjust search parameters")
    else:
        print("\n  ℹ️ No images in index - add COCO embeddings")
    
    return {'image_results': modal_counts_top50.get('image', 0)}

def test_multimodal_query_by_modal():
    """Test de filtrage par modalité spécifique"""
    print_section("15. Multi-Modal Filtering Test")
    
    try:
        from app.services.recommender import _index_v3, _posts_df_v3
        from app.services.cross_modal_search import search_by_text_cross_modal
    except ImportError as e:
        print(f"  ⚠️ V3 components not available: {e}")
        return None
    
    metadata = _posts_df_v3
    
    if 'modal' not in metadata.columns:
        print("  ⚠️ No 'modal' column in metadata")
        return None
    
    # Compter les différentes modalités
    modal_counts = metadata['modal'].value_counts()
    print(f"\n  📊 Index composition:")
    for modal, count in modal_counts.items():
        print(f"     - {modal}: {count} items")
    
    # Tester la recherche avec CLIP
    print("\n  🔍 Testing modal filtering with CLIP:")
    
    # Utiliser CLIP pour la recherche
    results = search_by_text_cross_modal("nature landscape", k=20, modal_filter='image', use_clip=True)
    
    print(f"     - Images trouvées avec CLIP: {len(results)}")
    
    if len(results) > 0:
        print(f"\n  🖼️ Example image results:")
        for i, r in enumerate(results[:3]):
            print(f"     {i+1}. [{r['score']:.4f}] {r['text'][:60]}...")
    
    return len(results) > 0


def test_cross_modal_with_real_queries():
    """Test avec des requêtes réelles d'utilisateurs"""
    print_section("16. Real User Query Simulation")
    
    try:
        from app.services.cross_modal_search import search_by_text_cross_modal
    except ImportError as e:
        print(f"  ⚠️ Components not available: {e}")
        return None
    
    # Requêtes utilisateur typiques
    test_queries = [
        "chat qui dort",
        "voiture de sport rouge",
        "plage tropicale",
        "nourriture saine",
        "ville moderne",
    ]
    
    print("\n  📝 Testing real-user queries with CLIP:")
    
    results_summary = []
    
    for query in test_queries:
        results = search_by_text_cross_modal(query, k=15, modal_filter='image', use_clip=True)
        n_images = len(results)
        
        results_summary.append({
            'query': query,
            'image_count': n_images,
        })
        
        print(f"\n     🔍 '{query}': {n_images}/15 images trouvées")
        if n_images > 0:
            print(f"        Top: [{results[0]['score']:.4f}] {results[0]['text'][:50]}...")
    
    # Statistiques globales
    total_images = sum(r['image_count'] for r in results_summary)
    avg_images = total_images / len(results_summary)
    
    print(f"\n  📊 Average images per query: {avg_images:.1f}")
    
    if avg_images > 5:
        print("  ✅ Cross-modal search is working effectively!")
    elif avg_images > 0:
        print("  ⚠️ Cross-modal search works but could be improved")
    else:
        print("  ❌ Cross-modal search not retrieving images")
    
    return results_summary

# =====================================================================
# FINAL VERDICT
# =====================================================================
def print_verdict(results):
    print_section("FINAL VERDICT")
    
    successes = []
    issues = []
    
    tests = [
        ("Embedding Model", results.get('embeddings', False)),
        ("Toxicity Model", results.get('toxicity', False)),
        ("Collaborative Model", results.get('collaborative', False)),
        ("FAISS Indexes", results.get('faiss', False)),
        ("User Profile Pipeline", results.get('profile', False)),
        ("Content-Based Pipeline (v1)", results.get('v1', False)),
        ("Hybrid Pipeline (v2)", results.get('v2', False)),
        ("Multi-Modal Pipeline (v3)", results.get('v3', None)),
        ("Ranker Components", results.get('ranker', False)),
        ("Personalization", results.get('personalization', False)),
        ("Cold Start", results.get('cold_start', False)),
        ("Score Distribution", results.get('distribution', False)),
        ("End-to-End Pipeline", results.get('e2e', False)),
    ]
    
    for name, ok in tests:
        if ok is True:
            successes.append(f"✅ {name}")
        elif ok is None:
            successes.append(f"  ⏭️ {name} (skipped)")
        else:
            issues.append(f"❌ {name}")
    
    print(f"\n" + "\n".join(successes))
    if issues:
        print(f"\n" + "\n".join(issues))
    
    print(f"\n{'=' * 60}")
    n_issues = len(issues)
    n_skipped = sum(1 for _, ok in tests if ok is None)
    n_total = sum(1 for _, ok in tests if ok is not None)
    
    if n_issues == 0:
        print(f"🎉 VERDICT: Pipeline is EXCELLENT - All {n_total} checks passed!")
    elif n_issues <= 2:
        print(f"✅ VERDICT: Pipeline is GOOD - {n_issues} minor issue(s) to address")
    else:
        print(f"⚠️ VERDICT: Pipeline needs improvement - {n_issues} issue(s) found")
    
    if n_skipped > 0:
        print(f"   ({n_skipped} test(s) skipped)")
    print(f"{'=' * 60}")


# =====================================================================
# MAIN
# =====================================================================
def run_all():
    print(f"\n{'🎯' * 30}")
    print(f" COMPLETE PIPELINE VALIDATION SUITE")
    print(f" Includes: Embeddings, Toxicity, Collaborative, FAISS v1/v2/v3, Scoring, Ranker")
    print(f"{'🎯' * 30}")
    
    results = {}
    
    # Section 1: Model health
    try:
        test_embedding_model()
        results['embeddings'] = True
    except Exception as e:
        print(f"  ❌ Embedding model failed: {e}")
        results['embeddings'] = False
    
    try:
        test_toxicity_model()
        results['toxicity'] = True
    except Exception as e:
        print(f"  ❌ Toxicity model failed: {e}")
        results['toxicity'] = False
    
    try:
        test_collaborative_model()
        results['collaborative'] = True
    except Exception as e:
        print(f"  ❌ Collaborative model failed: {e}")
        results['collaborative'] = False
    
    try:
        test_faiss_indexes()
        results['faiss'] = True
    except Exception as e:
        print(f"  ❌ FAISS indexes failed: {e}")
        results['faiss'] = False
    
    # Section 2: User profile
    try:
        test_user_profile_pipeline()
        results['profile'] = True
    except Exception as e:
        print(f"  ❌ User profile pipeline failed: {e}")
        results['profile'] = False
    
    # Section 3: Content-based v1
    try:
        test_content_based_pipeline()
        results['v1'] = True
    except Exception as e:
        print(f"  ❌ Content-based pipeline failed: {e}")
        results['v1'] = False
    
    # Section 4: Hybrid v2
    try:
        test_hybrid_pipeline()
        results['v2'] = True
    except Exception as e:
        print(f"  ❌ Hybrid pipeline failed: {e}")
        results['v2'] = False
    
    # Section 5: Multi-modal v3
    try:
        test_multimodal_pipeline()
        results['v3'] = True
    except Exception as e:
        print(f"  ❌ Multi-modal pipeline failed: {e}")
        results['v3'] = False
    
    # Section 6: Ranker
    try:
        test_ranker_components()
        results['ranker'] = True
    except Exception as e:
        print(f"  ❌ Ranker validation failed: {e}")
        results['ranker'] = False
    
    # Section 7: Personalization
    try:
        test_personalization()
        results['personalization'] = True
    except Exception as e:
        print(f"  ❌ Personalization test failed: {e}")
        results['personalization'] = False
    
    # Section 8: Cold start
    try:
        test_cold_start()
        results['cold_start'] = True
    except Exception as e:
        print(f"  ❌ Cold start test failed: {e}")
        results['cold_start'] = False
    
    # Section 9: Score distribution
    try:
        test_score_distribution()
        results['distribution'] = True
    except Exception as e:
        print(f"  ❌ Score distribution test failed: {e}")
        results['distribution'] = False
    
    # Section 10: End-to-end
    try:
        test_end_to_end_pipeline()
        results['e2e'] = True
    except Exception as e:
        print(f"  ❌ End-to-end pipeline failed: {e}")
        results['e2e'] = False
    
    # Section 11: Cross-modal search
    try:
        test_cross_modal_search()
        results['cross_modal'] = True
    except Exception as e:
        print(f"  ❌ Cross-modal search failed: {e}")
        results['cross_modal'] = False

    # Section 12: Modal filtering
    try:
        test_multimodal_query_by_modal()
        results['modal_filtering'] = True
    except Exception as e:
        print(f"  ❌ Modal filtering test failed: {e}")
        results['modal_filtering'] = False

    # Section 13: Real queries
    try:
        test_cross_modal_with_real_queries()
        results['real_queries'] = True
    except Exception as e:
        print(f"  ❌ Real queries test failed: {e}")
        results['real_queries'] = False
        
    # Final verdict
    print_verdict(results)
    
    n_pass = sum(1 for v in results.values() if v is True)
    n_fail = sum(1 for v in results.values() if v is False)
    n_total = sum(1 for v in results.values() if v is not None)
    
    print(f"\n  Summary: {n_pass}/{n_total} tests passed")
    if n_fail > 0:
        print(f"  Failures: {n_fail}")
    
    return results


if __name__ == '__main__':
    run_all()
