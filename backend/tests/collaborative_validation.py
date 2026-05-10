# tests/collaborative_validation.py
import sys
sys.path.insert(0, '.')

import json
import pickle
import numpy as np
from app.models.collaborative import (
    get_cf_scores, 
    get_top_cf_articles,
    get_user_recommendations_with_scores,
    _load_model
)

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_cold_start():
    """Test cold start for unknown user"""
    scores = get_cf_scores('unknown_user_xyz', [1, 2, 3])
    assert all(s == 0.5 for s in scores.values())
    print("✅ Cold start test passed")
    return True


def test_collaborative_model_performance():
    """Test the collaborative model directly (bypassing FAISS and scoring)"""
    print_section("1. Direct Collaborative Model Test")
    
    model, user2idx, art2idx, matrix = _load_model()
    
    print(f"\n📊 Model Info:")
    print(f"  - Users in model: {model.user_factors.shape[0]}")
    print(f"  - Items in model: {model.item_factors.shape[0]}")
    print(f"  - User2idx mapping size: {len(user2idx)}")
    print(f"  - Art2idx mapping size: {len(art2idx)}")
    
    # Test each user
    test_users = ['sim_user_17', 'sim_user_05', 'sim_user_04']
    
    results = {}
    for user in test_users:
        print(f"\n📊 Testing user: {user}")
        
        # Get recommendations with scores directly from collaborative model
        recs = get_user_recommendations_with_scores(user, n=10)
        
        if recs:
            scores = [s for _, s in recs]
            results[user] = {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'top_3': [(aid, s) for aid, s in recs[:3]]
            }
            
            print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
            print(f"  Score mean: {np.mean(scores):.4f}")
            print(f"  Score std: {np.std(scores):.4f}")
            print(f"  Top 3: {recs[:3]}")
            
            # Evaluate quality
            if max(scores) > 0.8:
                print("  ✅ Excellent: High scores detected!")
            elif max(scores) > 0.7:
                print("  ✅ Good: Scores are well distributed")
            elif max(scores) > 0.6:
                print("  ⚠️ Moderate: Scores could be higher")
            else:
                print("  ❌ Poor: Scores too low")
        else:
            print("  ❌ No recommendations found")
            results[user] = None
    
    return results


def test_collaborative_vs_user_comparison():
    """Compare recommendations across users to verify personalization"""
    print_section("2. Personalization Verification")
    
    users = ['sim_user_17', 'sim_user_05', 'sim_user_04']
    user_recs = {}
    
    for user in users:
        recs = get_top_cf_articles(user, n=10)
        user_recs[user] = set(recs)
        print(f"\n{user}:")
        print(f"  Top 10 articles: {recs[:5]}...")
    
    # Check if users have different recommendations
    if len(user_recs) >= 2:
        users_list = list(user_recs.keys())
        overlap_17_05 = len(user_recs[users_list[0]] & user_recs[users_list[1]])
        overlap_17_04 = len(user_recs[users_list[0]] & user_recs[users_list[2]])
        overlap_05_04 = len(user_recs[users_list[1]] & user_recs[users_list[2]])
        
        print(f"\n📊 Recommendation Overlap:")
        print(f"  sim_user_17 vs sim_user_05: {overlap_17_05}/10 common")
        print(f"  sim_user_17 vs sim_user_04: {overlap_17_04}/10 common")
        print(f"  sim_user_05 vs sim_user_04: {overlap_05_04}/10 common")
        
        if max(overlap_17_05, overlap_17_04, overlap_05_04) < 5:
            print("\n✅ Excellent: Users get very different recommendations!")
        elif max(overlap_17_05, overlap_17_04, overlap_05_04) < 8:
            print("\n✅ Good: Recommendations are personalized")
        else:
            print("\n⚠️ Warning: Users are getting similar recommendations")
    
    return user_recs


def test_score_statistics():
    """Analyze score statistics across all users"""
    print_section("3. Score Statistics Analysis")
    
    model, user2idx, art2idx, matrix = _load_model()
    
    all_scores = []
    user_stats = {}
    
    for user in list(user2idx.keys())[:10]:  # Test first 10 users
        recs = get_user_recommendations_with_scores(user, n=20)
        if recs:
            scores = [s for _, s in recs]
            all_scores.extend(scores)
            user_stats[user] = {
                'max': max(scores),
                'min': min(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
    
    if all_scores:
        print(f"\n📊 Global Statistics (over all users):")
        print(f"  Overall min score: {min(all_scores):.4f}")
        print(f"  Overall max score: {max(all_scores):.4f}")
        print(f"  Overall mean: {np.mean(all_scores):.4f}")
        print(f"  Overall std: {np.std(all_scores):.4f}")
        
        # Score distribution
        high = sum(1 for s in all_scores if s >= 0.7)
        medium = sum(1 for s in all_scores if 0.3 <= s < 0.7)
        low = sum(1 for s in all_scores if s < 0.3)
        
        print(f"\n📊 Score Distribution:")
        print(f"  High (>=0.7): {high} ({high/len(all_scores)*100:.1f}%)")
        print(f"  Medium (0.3-0.7): {medium} ({medium/len(all_scores)*100:.1f}%)")
        print(f"  Low (<0.3): {low} ({low/len(all_scores)*100:.1f}%)")
        
        # Evaluation
        if max(all_scores) >= 0.8:
            print("\n✅ Excellent: Model produces strong recommendations (>0.8)")
        elif max(all_scores) >= 0.7:
            print("\n✅ Good: Model produces solid recommendations (>0.7)")
        elif max(all_scores) >= 0.6:
            print("\n⚠️ Acceptable: Scores are moderate, could be higher")
        else:
            print("\n❌ Poor: Scores are too low")
            
        if low > 0:
            print("✅ Good: Model produces low scores for poor matches")
    else:
        print("❌ No scores found to analyze")
    
    return user_stats


def test_article_embedding_quality():
    """Test the quality of article embeddings"""
    print_section("4. Article Embedding Quality")
    
    model, user2idx, art2idx, matrix = _load_model()
    
    # Get item factors
    item_factors = model.item_factors
    print(f"\n📊 Item Embeddings:")
    print(f"  Shape: {item_factors.shape}")
    print(f"  Norm range: [{np.min(np.linalg.norm(item_factors, axis=1)):.4f}, "
          f"{np.max(np.linalg.norm(item_factors, axis=1)):.4f}]")
    
    # Calculate pairwise similarities between first 100 items
    n_samples = min(100, item_factors.shape[0])
    sample_factors = item_factors[:n_samples]
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(sample_factors, axis=1, keepdims=True)
    normalized = sample_factors / (norms + 1e-8)
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)
    
    # Get statistics (excluding self-similarity)
    off_diagonal = similarity_matrix[~np.eye(n_samples, dtype=bool)]
    
    print(f"\n📊 Embedding Similarity Statistics:")
    print(f"  Mean similarity: {np.mean(off_diagonal):.4f}")
    print(f"  Std similarity: {np.std(off_diagonal):.4f}")
    print(f"  Max similarity (non-self): {np.max(off_diagonal):.4f}")
    print(f"  Min similarity: {np.min(off_diagonal):.4f}")
    
    if np.mean(off_diagonal) < 0.3:
        print("  ✅ Good: Embeddings are well distributed")
    else:
        print("  ⚠️ Warning: Embeddings may be too similar")


def test_user_embedding_quality():
    """Test the quality of user embeddings"""
    print_section("5. User Embedding Quality")
    
    model, user2idx, art2idx, matrix = _load_model()
    
    user_factors = model.user_factors
    print(f"\n📊 User Embeddings:")
    print(f"  Shape: {user_factors.shape}")
    print(f"  Norm range: [{np.min(np.linalg.norm(user_factors, axis=1)):.4f}, "
          f"{np.max(np.linalg.norm(user_factors, axis=1)):.4f}]")
    
    # Calculate pairwise similarities
    n_users = user_factors.shape[0]
    if n_users > 1:
        norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
        normalized = user_factors / (norms + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        off_diagonal = similarity_matrix[~np.eye(n_users, dtype=bool)]
        
        print(f"\n📊 User Similarity Statistics:")
        print(f"  Mean similarity: {np.mean(off_diagonal):.4f}")
        print(f"  Min similarity: {np.min(off_diagonal):.4f}")
        print(f"  Max similarity: {np.max(off_diagonal):.4f}")
        
        if np.mean(off_diagonal) < 0.5:
            print("  ✅ Good: Users are well differentiated")
        else:
            print("  ⚠️ Warning: Users may be too similar")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🎯" * 30)
    print(" COLLABORATIVE FILTERING VALIDATION SUITE")
    print("🎯" * 30)
    
    # Test 1: Cold start
    test_cold_start()
    
    # Test 2: Direct collaborative model
    collab_results = test_collaborative_model_performance()
    
    # Test 3: Personalization
    test_collaborative_vs_user_comparison()
    
    # Test 4: Score statistics
    test_score_statistics()
    
    # Test 5: Embedding quality
    test_article_embedding_quality()
    test_user_embedding_quality()
    
    # Final Summary
    print_section("FINAL SUMMARY")
    
    # Check if model is healthy
    model, user2idx, art2idx, matrix = _load_model()
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    issues = []
    successes = []
    
    # Check embedding norms
    user_norms = np.linalg.norm(user_factors, axis=1)
    item_norms = np.linalg.norm(item_factors, axis=1)
    
    if np.mean(user_norms) > 0.1:
        successes.append("✅ User embeddings are healthy")
    else:
        issues.append("⚠️ User embeddings have very low magnitude")
    
    if np.mean(item_norms) > 0.1:
        successes.append("✅ Item embeddings are healthy")
    else:
        issues.append("⚠️ Item embeddings have very low magnitude")
    
    # Check if we have recommendations
    test_user = list(user2idx.keys())[0]
    recs = get_top_cf_articles(test_user, n=5)
    if recs:
        successes.append(f"✅ Model produces recommendations for {test_user}")
    else:
        issues.append(f"⚠️ No recommendations for {test_user}")
    
    # Print results
    print("\n" + "\n".join(successes))
    if issues:
        print("\n" + "\n".join(issues))
    
    # Overall verdict
    print("\n" + "=" * 60)
    if len(issues) == 0:
        print("🎉 VERDICT: Model is EXCELLENT - Production Ready!")
    elif len(issues) <= 2:
        print("✅ VERDICT: Model is GOOD - Ready for production with minor tuning")
    else:
        print("⚠️ VERDICT: Model needs improvement - Check training data")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()