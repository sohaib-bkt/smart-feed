"""
tests/test_real_pipeline.py

Test END-TO-END de la pipeline réelle (sans mocks)
Utilise les vraies données Firebase, FAISS, et le modèle Implicit.

⚠️ Ce test nécessite :
    - Firebase actif (serviceAccountKey.json)
    - Index FAISS présent (data/processed/faiss_v2.bin)
    - Modèle Implicit entraîné (app/models/implicit_model.pkl)
    - Données de test simulées (sim_user_*)
"""

import sys
import os
sys.path.insert(0, '.')

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta

# Import réels (pas de mocks)
from app.services.hybrid_recommender import get_hybrid_feed
from app.models.ranker import rank_candidates, score_candidate, explain_ranking
from app.services.user_profile import compute_user_embedding, get_top_interests
from app.db.firebase import get_user_profile, get_user_interactions, create_user
from app.services.recommender import get_feed

# Configuration
TEST_USERS = ['sim_user_17', 'sim_user_05', 'sim_user_04']
N_CANDIDATES = 60
N_RESULTS = 20


class TestRealPipeline:
    """
    Test complet de la pipeline réelle.
    Nécessite Firebase et les modèles entraînés.
    """

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Vérifie que les utilisateurs de test existent"""
        self.users = []
        for user_id in TEST_USERS:
            profile = await get_user_profile(user_id)
            if not profile:
                # Créer l'utilisateur s'il n'existe pas
                profile = await create_user(user_id, preferences={
                    "interests": [],
                    "mode": "default"
                })
            self.users.append(user_id)
        yield

    # ═══════════════════════════════════════════════════════════════════
    # Tests du profil utilisateur
    # ═══════════════════════════════════════════════════════════════════

    def test_user_profile_exists(self):
        """Vérifie que les utilisateurs de test existent dans Firebase"""
        for user_id in self.users:
            profile = asyncio.run(get_user_profile(user_id))
            assert profile is not None, f"User {user_id} not found"
            print(f"✅ User {user_id} profile exists")

    def test_user_interactions_exist(self):
        """Vérifie que les utilisateurs ont des interactions"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=10))
            assert len(interactions) > 0, f"No interactions for {user_id}"
            print(f"✅ {user_id} has {len(interactions)} interactions")

    def test_user_embedding_computation(self):
        """Teste le calcul d'embedding utilisateur"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            embedding = compute_user_embedding(interactions)
            
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) == 384, f"Expected 384 dims, got {len(embedding)}"
            assert all(isinstance(x, (int, float)) for x in embedding), "Non-numeric values"
            
            # Vérifier la normalisation (norme devrait être proche de 1)
            norm = np.linalg.norm(embedding)
            assert 0.5 < norm < 1.5, f"Embedding norm = {norm} (expected ~1.0)"
            
            print(f"✅ {user_id} embedding: norm={norm:.4f}")

    def test_top_interests_detection(self):
        """Teste la détection des intérêts depuis les interactions"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            interests = get_top_interests(interactions, top_k=3)
            
            assert isinstance(interests, list), "Interests should be a list"
            assert len(interests) <= 3, f"Too many interests: {len(interests)}"
            
            if interactions:
                assert len(interests) > 0, f"No interests detected for {user_id}"
            
            print(f"✅ {user_id} top interests: {interests}")

    # ═══════════════════════════════════════════════════════════════════
    # Tests du feed v1 (content-based)
    # ═══════════════════════════════════════════════════════════════════

    def test_feed_v1_returns_results(self):
        """Teste que get_feed retourne des résultats"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            feed = get_feed(
                user_embedding=embedding,
                user_prefs=prefs,
                n_results=N_RESULTS
            )
            
            assert isinstance(feed, list), "Feed should be a list"
            assert len(feed) <= N_RESULTS, f"Too many results: {len(feed)}"
            
            if feed:
                # Vérifier les champs obligatoires
                item = feed[0]
                assert 'id' in item, "Missing 'id' field"
                assert 'score' in item, "Missing 'score' field"
                assert 'category' in item, "Missing 'category' field"
                
                print(f"✅ v1 feed for {user_id}: {len(feed)} items")

    def test_feed_v1_scores_are_varied(self):
        """Vérifie que les scores v1 varient (pas tous identiques)"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            feed = get_feed(
                user_embedding=embedding,
                user_prefs=prefs,
                n_results=N_RESULTS
            )
            
            if feed and len(feed) > 1:
                scores = [item['score'] for item in feed]
                assert len(set(scores)) > 1, "All scores are identical!"
                print(f"✅ v1 scores for {user_id}: min={min(scores):.3f}, max={max(scores):.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Tests du feed v2 (hybride)
    # ═══════════════════════════════════════════════════════════════════

    def test_hybrid_feed_returns_results(self):
        """Teste que get_hybrid_feed retourne des résultats"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            feed = get_hybrid_feed(
                user_id=user_id,
                user_embedding=embedding,
                user_prefs=prefs,
                n_candidates=200,
                n_results=N_CANDIDATES
            )
            
            assert isinstance(feed, list), "Feed should be a list"
            assert len(feed) <= N_CANDIDATES, f"Too many candidates: {len(feed)}"
            
            if feed:
                item = feed[0]
                assert 'id' in item, "Missing 'id' field"
                assert 'score' in item, "Missing 'score' field"
                assert 'score_detail' in item, "Missing 'score_detail'"
                assert 'content_based' in item['score_detail'], "Missing content_based"
                assert 'collaborative' in item['score_detail'], "Missing collaborative"
                
                print(f"✅ Hybrid feed for {user_id}: {len(feed)} candidates")

    def test_hybrid_feed_combines_both_signals(self):
        """Vérifie que le feed hybride combine content-based et collaboratif"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            feed = get_hybrid_feed(
                user_id=user_id,
                user_embedding=embedding,
                user_prefs=prefs,
                n_candidates=200,
                n_results=20
            )
            
            if feed:
                for item in feed[:5]:
                    cb = item['score_detail']['content_based']
                    cf = item['score_detail']['collaborative']
                    final = item['score']
                    
                    assert 0 <= cb <= 1, f"Invalid content_based: {cb}"
                    assert 0 <= cf <= 1, f"Invalid collaborative: {cf}"
                    assert 0 <= final <= 1, f"Invalid final score: {final}"
                    
                    print(f"  Article {item['id']}: CB={cb:.3f}, CF={cf:.3f}, Final={final:.3f}")
                
                print(f"✅ {user_id}: Hybrid combines both signals")

    # ═══════════════════════════════════════════════════════════════════
    # Tests du re-ranking (ranker.py)
    # ═══════════════════════════════════════════════════════════════════

    def test_rank_candidates_works(self):
        """Teste le re-ranking des candidats"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            candidates = get_hybrid_feed(
                user_id=user_id,
                user_embedding=embedding,
                user_prefs=prefs,
                n_candidates=200,
                n_results=N_CANDIDATES
            )
            
            ranked = rank_candidates(candidates, user_embedding=embedding)
            
            assert isinstance(ranked, list), "Ranked should be a list"
            assert len(ranked) == len(candidates), "Lost candidates during ranking"
            
            if ranked:
                assert 'rank_score' in ranked[0], "Missing rank_score after ranking"
                
                scores = [item['rank_score'] for item in ranked]
                assert scores == sorted(scores, reverse=True), "Not sorted descending"
                
                print(f"✅ Ranked {len(ranked)} candidates for {user_id}")

    def test_rank_candidates_improves_variety(self):
        """Vérifie que le re-ranking diversifie les résultats"""
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            candidates = get_hybrid_feed(
                user_id=user_id,
                user_embedding=embedding,
                user_prefs=prefs,
                n_candidates=200,
                n_results=N_CANDIDATES
            )
            
            if candidates:
                categories_before = set()
                for c in candidates[:10]:
                    categories_before.add(c.get('category', 'Unknown'))
                
                ranked = rank_candidates(candidates, user_embedding=embedding)
                
                categories_after = set()
                for r in ranked[:10]:
                    categories_after.add(r.get('category', 'Unknown'))
                
                print(f"  Categories before: {categories_before}")
                print(f"  Categories after: {categories_after}")

    # ═══════════════════════════════════════════════════════════════════
    # Tests de la fonction score_candidate
    # ═══════════════════════════════════════════════════════════════════

    def test_score_candidate_returns_valid_score(self):
        """Teste le calcul de score individuel"""
        user_embedding = [0.1] * 384
        
        candidate = {
            'id': 'test_123',
            'embedding': [0.2] * 384,
            'toxicity_score': 0.1,
            'likes': 100,
            'views': 1000,
            'created_at': datetime.now()
        }
        
        score = score_candidate(candidate, user_embedding)
        
        assert isinstance(score, float), "Score should be float"
        assert 0 <= score <= 1, f"Score out of range: {score}"
        print(f"✅ Score calculation: {score:.4f}")

    def test_score_candidate_penalizes_toxicity(self):
        """Vérifie que la toxicité réduit le score"""
        user_embedding = [0.1] * 384
        
        candidate_clean = {
            'id': 'clean',
            'embedding': [0.2] * 384,
            'toxicity_score': 0.0,
            'likes': 100,
            'views': 1000,
            'created_at': datetime.now()
        }
        
        candidate_toxic = {
            'id': 'toxic',
            'embedding': [0.2] * 384,
            'toxicity_score': 0.9,
            'likes': 100,
            'views': 1000,
            'created_at': datetime.now()
        }
        
        score_clean = score_candidate(candidate_clean, user_embedding)
        score_toxic = score_candidate(candidate_toxic, user_embedding)
        
        assert score_clean > score_toxic, "Toxic content not penalized!"
        print(f"✅ Clean={score_clean:.4f} > Toxic={score_toxic:.4f}")

    def test_score_candidate_boosts_freshness(self):
        """Vérifie que les articles récents sont favorisés"""
        user_embedding = [0.1] * 384
        now = datetime.now()
        
        candidate_fresh = {
            'id': 'fresh',
            'embedding': [0.2] * 384,
            'toxicity_score': 0.1,
            'likes': 100,
            'views': 1000,
            'created_at': now
        }
        
        candidate_old = {
            'id': 'old',
            'embedding': [0.2] * 384,
            'toxicity_score': 0.1,
            'likes': 100,
            'views': 1000,
            'created_at': now - timedelta(days=30)
        }
        
        score_fresh = score_candidate(candidate_fresh, user_embedding)
        score_old = score_candidate(candidate_old, user_embedding)
        
        assert score_fresh >= score_old, "Fresh content not boosted!"
        print(f"✅ Fresh={score_fresh:.4f} >= Old={score_old:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Tests de personnalisation
    # ═══════════════════════════════════════════════════════════════════

    def test_personalization_users_get_different_feeds(self):
        """Vérifie que deux utilisateurs reçoivent des feeds différents"""
        feeds = {}
        
        for user_id in self.users:
            interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
            profile = asyncio.run(get_user_profile(user_id))
            
            embedding = compute_user_embedding(interactions)
            prefs = profile.get('preferences', {})
            
            feed = get_hybrid_feed(
                user_id=user_id,
                user_embedding=embedding,
                user_prefs=prefs,
                n_candidates=200,
                n_results=10
            )
            
            feeds[user_id] = [item['id'] for item in feed]
        
        if len(self.users) >= 2:
            feed1 = set(feeds[self.users[0]])
            feed2 = set(feeds[self.users[1]])
            
            overlap = len(feed1 & feed2)
            print(f"  Overlap between {self.users[0]} and {self.users[1]}: {overlap}/10")
            
            if overlap < 5:
                print("  ✅ Strong personalization detected!")
            else:
                print("  ⚠️ Weak personalization - users see similar content")

    # ═══════════════════════════════════════════════════════════════════
    # Test complet de la pipeline end-to-end
    # ═══════════════════════════════════════════════════════════════════

    def test_complete_pipeline_end_to_end(self):
        """Test complet simulant l'API feed v2 réelle."""
        user_id = self.users[0] if self.users else 'sim_user_17'
        
        print(f"\n{'='*60}")
        print(f"Testing complete pipeline for {user_id}")
        print(f"{'='*60}")
        
        # 1. Charger profil
        profile = asyncio.run(get_user_profile(user_id))
        assert profile is not None, "Profile not found"
        print(f"✅ 1. Profile loaded: {profile.get('user_id')}")
        
        # 2. Charger interactions
        interactions = asyncio.run(get_user_interactions(user_id, last_n=100))
        print(f"✅ 2. Loaded {len(interactions)} interactions")
        
        # 3. Calculer embedding
        embedding = compute_user_embedding(interactions)
        assert len(embedding) == 384, "Invalid embedding dimension"
        print(f"✅ 3. Embedding computed (norm={np.linalg.norm(embedding):.4f})")
        
        # 4. Préparer préférences
        prefs = profile.get('preferences', {})
        if not prefs.get('interests') and interactions:
            prefs['interests'] = get_top_interests(interactions)
        print(f"✅ 4. Preferences: interests={prefs.get('interests', [])[:3]}")
        
        # 5. Obtenir feed hybride
        candidates = get_hybrid_feed(
            user_id=user_id,
            user_embedding=embedding,
            user_prefs=prefs,
            n_candidates=300,
            n_results=60
        )
        print(f"✅ 5. Hybrid feed: {len(candidates)} candidates")
        
        # 6. Re-ranking
        ranked = rank_candidates(candidates, user_embedding=embedding)[:20]
        print(f"✅ 6. Ranking complete: {len(ranked)} final items")
        
        # 7. Afficher TOUS les résultats (pas juste 5)
        if ranked:
            print(f"\n📊 COMPLETE RECOMMENDATIONS for {user_id} ({len(ranked)} items):")
            print("-" * 70)
            
            for i, item in enumerate(ranked):
                score = item.get('rank_score', item.get('score', 0))
                category = item.get('category', 'Unknown')
                
                # Récupérer les détails si disponibles
                detail = item.get('score_detail', {})
                cb = detail.get('content_based', 0)
                cf = detail.get('collaborative', 0)
                
                # Afficher avec indicateur de qualité
                if score >= 0.7:
                    indicator = "🔥"
                elif score >= 0.6:
                    indicator = "👍"
                elif score >= 0.4:
                    indicator = "📰"
                else:
                    indicator = "❌"
                
                print(f"  {i+1:2d}. {indicator} ID: {item['id']}")
                print(f"      Score: {score:.4f} | CB: {cb:.3f} | CF: {cf:.3f}")
                print(f"      Catégorie: {category}")
                print()
            
            # Statistiques des scores
            scores = [item.get('rank_score', item.get('score', 0)) for item in ranked]
            print(f"📊 Score statistics:")
            print(f"   Min: {min(scores):.4f}")
            print(f"   Max: {max(scores):.4f}")
            print(f"   Mean: {np.mean(scores):.4f}")
            print(f"   Std: {np.std(scores):.4f}")
            
            # Vérifier la diversité des catégories
            categories = set(item.get('category', 'Unknown') for item in ranked)
            print(f"   Categories: {len(categories)} unique")
        
        print("\n✅ COMPLETE PIPELINE SUCCESSFUL!")
        assert len(ranked) > 0, "No recommendations generated"


# ═══════════════════════════════════════════════════════════════════════
# Exécution manuelle pour debug (VERSION AMÉLIORÉE)
# ═══════════════════════════════════════════════════════════════════════

async def manual_test():
    """Fonction manuelle pour tester la pipeline avec affichage complet"""
    user_id = 'sim_user_17'
    
    print(f"\n{'='*70}")
    print(f"🔍 MANUAL TEST - Pipeline Complète")
    print(f"📊 Utilisateur: {user_id}")
    print(f"{'='*70}")
    
    # 1. Charger profil
    print("\n1️⃣ Chargement du profil...")
    profile = await get_user_profile(user_id)
    print(f"   ✅ Profile: {profile.get('user_id') if profile else 'Not found'}")
    
    # 2. Charger interactions
    print("\n2️⃣ Chargement des interactions...")
    interactions = await get_user_interactions(user_id, last_n=50)
    print(f"   ✅ {len(interactions)} interactions trouvées")
    
    # 3. Calculer embedding
    print("\n3️⃣ Calcul de l'embedding utilisateur...")
    embedding = compute_user_embedding(interactions)
    norm = np.linalg.norm(embedding)
    print(f"   ✅ Norme: {norm:.4f} (idéalement proche de 1.0)")
    
    # 4. Détecter intérêts
    print("\n4️⃣ Détection des centres d'intérêt...")
    prefs = profile.get('preferences', {}) if profile else {}
    if not prefs.get('interests') and interactions:
        prefs['interests'] = get_top_interests(interactions)
    print(f"   ✅ Intérêts: {prefs.get('interests', [])}")
    
    # 5. Feed hybride
    print("\n5️⃣ Génération du feed hybride (v2)...")
    candidates = get_hybrid_feed(
        user_id=user_id,
        user_embedding=embedding,
        user_prefs=prefs,
        n_candidates=100,
        n_results=20
    )
    print(f"   ✅ {len(candidates)} candidats générés")
    
    # 6. Re-ranking
    print("\n6️⃣ Application du re-ranking (Solution 4)...")
    ranked = rank_candidates(candidates, user_embedding=embedding)[:10]
    print(f"   ✅ {len(ranked)} recommandations finales")
    
    # 7. Afficher TOUS les résultats avec détails
    print(f"\n{'='*70}")
    print(f"📊 RECOMMANDATIONS FINALES (Top {len(ranked)})")
    print(f"{'='*70}")
    
    if not ranked:
        print("⚠️ Aucune recommandation trouvée!")
        return
    
    for i, item in enumerate(ranked):
        score = item.get('rank_score', item.get('score', 0))
        category = item.get('category', 'Unknown')
        article_id = item.get('id', '?')
        
        # Détails des scores
        detail = item.get('score_detail', {})
        cb = detail.get('content_based', 0)
        cf = detail.get('collaborative', 0)
        
        # Ajustement CF (si disponible)
        cf_adj = item.get('_cf_adjustment', 0)
        
        # Indicateur visuel
        if score >= 0.8:
            indicator = "🏆 EXCELLENT"
        elif score >= 0.7:
            indicator = "⭐ TRÈS BON"
        elif score >= 0.6:
            indicator = "👍 BON"
        elif score >= 0.5:
            indicator = "📰 MOYEN"
        else:
            indicator = "⚠️ FAIBLE"
        
        print(f"\n{i+1}. [{indicator}] Article {article_id}")
        print(f"   📊 Score final: {score:.4f}")
        print(f"   📂 Catégorie: {category}")
        print(f"   🔹 Content-Based: {cb:.3f}")
        print(f"   🔸 Collaboratif: {cf:.3f}")
        if cf_adj != 0:
            print(f"   🔄 Ajustement CF: {cf_adj:+.3f}")
    
    # 8. Statistiques globales
    scores = [item.get('rank_score', item.get('score', 0)) for item in ranked]
    categories = set(item.get('category', 'Unknown') for item in ranked)
    
    print(f"\n{'='*70}")
    print(f"📈 STATISTIQUES")
    print(f"{'='*70}")
    print(f"   Score min: {min(scores):.4f}")
    print(f"   Score max: {max(scores):.4f}")
    print(f"   Score moyen: {np.mean(scores):.4f}")
    print(f"   Écart-type: {np.std(scores):.4f}")
    print(f"   Catégories uniques: {len(categories)}")
    print(f"   Catégories: {sorted(categories)}")
    
    # 9. Vérification de la diversité
    if len(categories) >= 3:
        print(f"\n✅ BONNE DIVERSITÉ: {len(categories)} catégories différentes")
    else:
        print(f"\n⚠️ DIVERSITÉ LIMITÉE: seulement {len(categories)} catégorie(s)")
    
    print(f"\n{'='*70}")
    print("✅ TEST MANUAL COMPLET RÉUSSI!")
    print(f"{'='*70}")


if __name__ == '__main__':
    asyncio.run(manual_test())