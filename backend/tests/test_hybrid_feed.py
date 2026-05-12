# tests/test_hybrid_feed.py
import sys
sys.path.insert(0, '.')

import numpy as np
from app.services.hybrid_recommender import get_hybrid_feed
from app.services.user_profile import compute_user_embedding
from app.db.firebase import get_user_profile, get_user_interactions
import asyncio


async def test_hybrid():
    user_id = "sim_user_17"
    
    print(f"Testing hybrid feed for {user_id}")
    print("-" * 50)
    
    # Get user data
    profile = await get_user_profile(user_id)
    interactions = await get_user_interactions(user_id, last_n=100)
    
    print(f"Found {len(interactions)} interactions")
    print(f"User profile: {profile.get('interests', [])[:3]}...")
    
    # Compute embedding - FIXED: handle both list and array returns
    if interactions:
        embedding_result = compute_user_embedding(interactions)
        # Check if it's already a list or needs conversion
        if hasattr(embedding_result, 'tolist'):
            user_embedding = embedding_result.tolist()
        else:
            user_embedding = embedding_result  # Already a list
    else:
        user_embedding = [0.0] * 384
    
    print(f"User embedding shape: {len(user_embedding)}")
    
    # Get hybrid feed
    feed = get_hybrid_feed(
        user_id=user_id,
        user_embedding=user_embedding,
        user_prefs=profile or {},
        n_candidates=200,
        n_results=10
    )
    
    print(f"\n{'='*60}")
    print(f"Hybrid Feed for {user_id}")
    print(f"{'='*60}")
    
    if feed:
        for i, item in enumerate(feed[:5]):
            print(f"\n{i+1}. ID: {item['id']}")
            print(f"   Headline: {item['headline'][:80]}...")
            print(f"   Category: {item['category']}")
            print(f"   Final Score: {item['score']:.4f}")
            if 'score_detail' in item:
                print(f"   Content-based: {item['score_detail'].get('content_based', 'N/A')}")
                print(f"   Collaborative: {item['score_detail'].get('collaborative', 'N/A')}")
            print(f"   {item['explanation']}")
    else:
        print("No recommendations found")
    
    return feed

if __name__ == '__main__':
    asyncio.run(test_hybrid())