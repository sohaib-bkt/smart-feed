# tests/test_collaborative.py
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

def test_cold_start():
    """Test cold start for unknown user"""
    scores = get_cf_scores('unknown_user_xyz', [1, 2, 3])
    assert all(s == 0.5 for s in scores.values())
    print("✅ Cold start test passed")


def test_existing_users():
    """Test with actual users from the model"""
    model, user2idx, art2idx, matrix = _load_model()
    
    # Get actual users from the model
    print(f"\n📊 Model Info:")
    print(f"  - Users in model: {model.user_factors.shape[0]}")
    print(f"  - Items in model: {model.item_factors.shape[0]}")
    print(f"  - User2idx mapping size: {len(user2idx)}")
    print(f"  - Art2idx mapping size: {len(art2idx)}")
    
    # Get first few actual users
    actual_users = list(user2idx.keys())[:5]
    print(f"\n  Actual users in mapping: {actual_users}")
    
    # Test each actual user
    for user in actual_users:
        print(f"\n📊 Testing user: {user}")
        
        # Get top articles
        top_articles = get_top_cf_articles(user, n=5)
        print(f"  Top 5 articles: {top_articles}")
        
        # Test scores for specific articles
        if top_articles:
            test_articles = top_articles[:3]
            scores = get_cf_scores(user, test_articles)
            print(f"  Scores for {test_articles}: {scores}")
            
            # Verify scores are not all 0.5
            score_values = list(scores.values())
            if any(s != 0.5 for s in score_values):
                print("  ✅ Personalized scores detected!")
            else:
                print("  ⚠️ All scores are 0.5 - personalization not working")
        else:
            print("  ⚠️ No recommendations found")


def test_specific_articles():
    """Test with specific article IDs"""
    model, user2idx, art2idx, matrix = _load_model()
    
    # Get first user
    if user2idx:
        user = list(user2idx.keys())[0]
        print(f"\n📊 Testing user: {user}")
        
        # Get some article IDs from the mapping
        article_ids = list(art2idx.keys())[:5]
        article_ids_int = [int(aid) for aid in article_ids]
        print(f"  Testing articles: {article_ids_int}")
        
        scores = get_cf_scores(user, article_ids_int)
        print(f"  Scores: {scores}")
        
        # Check if scores vary
        score_values = list(scores.values())
        if len(set(score_values)) > 1:
            print("  ✅ Scores vary - model working!")
        elif score_values and score_values[0] == 0.5:
            print("  ⚠️ All scores are 0.5 - check model loading")
        else:
            print(f"  Scores: {score_values}")


def test_recommendations_with_scores():
    """Test recommendations with scores"""
    model, user2idx, art2idx, matrix = _load_model()
    
    if not user2idx:
        print("❌ No users found in mapping")
        return
    
    user = list(user2idx.keys())[0]
    print(f"\n📊 Getting recommendations for: {user}")
    
    recs = get_user_recommendations_with_scores(user, n=5)
    print(f"  Recommendations: {recs}")
    
    if recs:
        # Check if scores are meaningful
        scores = [score for _, score in recs]
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        
        if max(scores) > 0.6:
            print("  ✅ Good score range detected!")
        else:
            print("  ⚠️ Scores are low - model may need more training")


def find_working_user():
    """Find a user that actually has recommendations"""
    model, user2idx, art2idx, matrix = _load_model()
    
    print("\n🔍 Searching for user with recommendations...")
    
    for user in list(user2idx.keys())[:10]:
        top = get_top_cf_articles(user, n=3)
        if top:
            print(f"  ✅ User {user} has recommendations: {top}")
            return user
    
    print("  ❌ No users found with recommendations")
    return None


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Collaborative Filtering")
    print("=" * 60)
    
    # First, find a working user
    working_user = find_working_user()
    
    # Run tests
    test_cold_start()
    test_existing_users()
    test_specific_articles()
    test_recommendations_with_scores()
    
    # If we found a working user, test specifically with them
    if working_user:
        print(f"\n📊 Detailed test with working user: {working_user}")
        recs = get_user_recommendations_with_scores(working_user, n=10)
        print(f"  Top 10 recommendations: {recs[:5]}...")
        
        # Calculate score statistics
        if recs:
            scores = [s for _, s in recs]
            print(f"  Score stats - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")