import pandas as pd
import numpy as np
import asyncio
import random
import sys
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, '.')
from app.db.firebase import log_interaction, create_user, get_db

# Load data
try:
    df = pd.read_parquet('data/raw/huffpost.parquet')
    print(f"✅ Loaded {len(df)} articles from huffpost.parquet")
except FileNotFoundError:
    print("❌ Error: data/raw/huffpost.parquet not found!")
    print("Please run preprocessing scripts first.")
    sys.exit(1)

N_USERS = 20
CATEGORIES = df['category'].unique().tolist()
print(f"📂 Available categories: {CATEGORIES[:5]}... ({len(CATEGORIES)} total)")

print("\n🔢 Loading pre-computed embeddings...")
embeddings = np.load('data/processed/huffpost_embeddings.npy')
article_embeddings = {str(i): embeddings[i].tolist() for i in range(len(embeddings))}
print(f"✅ Loaded {len(article_embeddings)} article embeddings")

class UserBehavior:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.activity_level = random.choice(['low', 'medium', 'high', 'very_high'])
        
        # Ensure we don't select more categories than available
        num_preferred = min(random.randint(2, 5), len(CATEGORIES))
        self.preferred_categories = random.sample(CATEGORIES, k=num_preferred)
        
        # Disliked categories from non-preferred
        non_preferred = [c for c in CATEGORIES if c not in self.preferred_categories]
        if non_preferred:
            num_disliked = min(random.randint(0, 2), len(non_preferred))
            self.disliked_categories = random.sample(non_preferred, k=num_disliked)
        else:
            self.disliked_categories = []
        
        self.engagement_style = random.choices(
            ['scroller', 'commenter', 'saver', 'minimalist'],
            weights=[0.35, 0.15, 0.15, 0.35]
        )[0]
        
        self.preferred_action_probs = self._get_action_probs()
    
    def _get_action_probs(self):
        styles = {
            'scroller':   {'like': 0.25, 'watch_full': 0.30, 'watch_50': 0.25, 'skip': 0.15, 'comment': 0.03, 'share': 0.02},
            'commenter':  {'like': 0.20, 'watch_full': 0.25, 'watch_50': 0.20, 'skip': 0.15, 'comment': 0.15, 'share': 0.05},
            'saver':      {'like': 0.30, 'watch_full': 0.35, 'watch_50': 0.15, 'skip': 0.10, 'comment': 0.03, 'share': 0.07},
            'minimalist': {'like': 0.10, 'watch_full': 0.15, 'watch_50': 0.20, 'skip': 0.40, 'comment': 0.05, 'share': 0.10},
        }
        return styles.get(self.engagement_style, styles['scroller'])
    
    def get_action(self, category: str):
        # Strong negative reaction to disliked categories
        if category in self.disliked_categories:
            return random.choices(['skip', 'hide'], weights=[0.7, 0.3])[0]
        
        # Positive engagement with preferred categories
        if category in self.preferred_categories:
            actions = list(self.preferred_action_probs.keys())
            weights = list(self.preferred_action_probs.values())
            # Boost positive actions for preferred content
            if random.random() < 0.3:  # 30% chance to upgrade to more positive action
                if 'like' in actions:
                    return 'like'
                elif 'watch_full' in actions:
                    return 'watch_full'
            return random.choices(actions, weights=weights)[0]
        
        # Neutral categories
        return random.choices(
            ['watch_50', 'skip', 'like', 'watch_full'],
            weights=[0.35, 0.35, 0.20, 0.10]
        )[0]
    
    def get_num_interactions(self):
        levels = {
            'low': (15, 40),
            'medium': (40, 80),
            'high': (80, 150),
            'very_high': (150, 300)
        }
        min_val, max_val = levels.get(self.activity_level, (30, 60))
        return random.randint(min_val, max_val)


def generate_session_times(start_date, num_sessions):
    """Generate realistic session times with daily patterns"""
    sessions = []
    for _ in range(num_sessions):
        # Spread sessions over last 30 days
        days_offset = random.randint(0, 30)
        
        # Realistic time-of-day distribution (peaks: morning 8-10, evening 19-22)
        hour = random.choices(
            list(range(24)),
            weights=[1,1,1,1,1,1,    # 0-5 (night)
                    5,8,10,8,5,      # 6-10 (morning peak)
                    3,3,4,5,5,       # 11-15 (afternoon)
                    6,8,10,9,6,      # 16-20 (evening peak)
                    3,2,1]           # 21-23 (late evening)
        )[0]
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        session_time = start_date + timedelta(days=days_offset, hours=hour, minutes=minute, seconds=second)
        sessions.append(session_time)
    
    return sorted(sessions)


async def create_user_with_profile(user_id: str, behavior: UserBehavior):
    """Create user profile in Firebase"""
    preferences = {
        "interests": behavior.preferred_categories,
        "disliked_categories": behavior.disliked_categories,
        "toxicity_threshold": round(random.uniform(0.2, 0.5), 2),
        "default_mode": random.choice(['default', 'fun', 'serious', 'fresh']),
        "content_type": random.choice(['all', 'articles', 'mixed']),
        "activity_level": behavior.activity_level,
        "engagement_style": behavior.engagement_style,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        await create_user(user_id, preferences=preferences)
        return True
    except Exception as e:
        print(f"  ⚠️ Error creating user {user_id}: {e}")
        return False


async def clear_existing_user_data(user_id: str):
    """Clear existing interactions for a user to avoid duplicates"""
    try:
        db = get_db()
        interactions = db.collection('interactions').where('user_id', '==', user_id).stream()
        
        batch = db.batch()
        count = 0
        for doc in interactions:
            batch.delete(doc.reference)
            count += 1
            if count % 500 == 0:
                batch.commit()
                batch = db.batch()
        
        if count % 500 != 0:
            batch.commit()
        
        if count > 0:
            print(f"  🗑️ Cleared {count} existing interactions for {user_id}")
        return count
    except Exception as e:
        print(f"  ⚠️ Could not clear existing data for {user_id}: {e}")
        return 0


async def simulate():
    print(f'\n{"="*60}')
    print(f'🎯 Starting realistic user simulation')
    print(f'{"="*60}')
    print(f'📊 Config: {N_USERS} users, {len(CATEGORIES)} categories')
    
    # Create users with realistic behaviors
    print(f'\n👥 Generating {N_USERS} users with realistic behaviors...')
    users = [UserBehavior(f'sim_user_{i:02d}') for i in range(N_USERS)]
    
    # Create user profiles
    print(f'\n📝 Creating user profiles in Firebase...')
    for i, user in enumerate(users):
        await create_user_with_profile(user.user_id, user)
        if (i + 1) % 10 == 0:
            print(f'  ✅ Created {i+1}/{N_USERS} profiles')
    
    # Generate interactions
    print(f'\n🔄 Generating interactions...')
    base_time = datetime.now() - timedelta(days=30)
    total_interactions = 0
    batch_size = 0
    
    for user in users:
        num_interactions = user.get_num_interactions()
        num_sessions = max(1, num_interactions // 8)  # ~8 interactions per session
        session_times = generate_session_times(base_time, num_sessions)
        
        # Generate candidate articles
        candidate_articles = []
        for _ in range(num_interactions):
            # Category selection with realistic distribution
            rand = random.random()
            if rand < 0.70:  # 70% preferred categories
                category = random.choice(user.preferred_categories)
            elif rand < 0.85:  # 15% neutral categories
                category = random.choice([c for c in CATEGORIES 
                                         if c not in user.preferred_categories 
                                         and c not in user.disliked_categories] or CATEGORIES)
            else:  # 15% disliked categories (testing boundaries)
                category = random.choice(user.disliked_categories) if user.disliked_categories else random.choice(CATEGORIES)
            
            # Get articles from this category
            articles = df[df['category'] == category]
            if len(articles) > 0:
                article = articles.sample(1).iloc[0]
                candidate_articles.append((article, category))
        
        # Log interactions with realistic timing
        for idx, (article, category) in enumerate(candidate_articles):
            # Assign to a session
            session_idx = min(idx // 8, len(session_times) - 1)
            timestamp = session_times[session_idx] + timedelta(minutes=random.randint(0, 45))
            
            # Ensure timestamp is not in future
            if timestamp > datetime.now():
                timestamp = datetime.now() - timedelta(minutes=random.randint(1, 60))
            
            action = user.get_action(category)
            post_id = str(article.name)
            
            interaction = {
                'user_id': user.user_id,
                'post_id': post_id,
                'action': action,
                'post_text': str(article['text'])[:500] if pd.notna(article.get('text')) else '',
                'category': category,
                'embedding': article_embeddings.get(post_id, [0.0] * 384),
                'timestamp': timestamp.isoformat(),
            }
            
            try:
                await log_interaction(interaction)
                total_interactions += 1
                batch_size += 1
            except Exception as e:
                print(f"  ⚠️ Error logging interaction for {user.user_id}: {e}")
        
        # Progress update
        if int(user.user_id.split('_')[-1]) % 10 == 0:
            print(f'  📈 Progress: {total_interactions} interactions so far...')
    
    # Final summary
    print(f'\n{"="*60}')
    print(f'✅ Simulation Complete!')
    print(f'{"="*60}')
    print(f'📊 Statistics:')
    print(f'  • Users created: {N_USERS}')
    print(f'  • Total interactions: {total_interactions}')
    print(f'  • Avg interactions/user: {total_interactions // N_USERS}')
    
    # User type distribution
    styles = defaultdict(int)
    levels = defaultdict(int)
    for user in users:
        styles[user.engagement_style] += 1
        levels[user.activity_level] += 1
    
    print(f'\n👥 User Persona Distribution:')
    print(f'  • Engagement styles: {dict(styles)}')
    print(f'  • Activity levels: {dict(levels)}')
    
    # Category preference stats
    fav_cats = defaultdict(int)
    for user in users:
        for cat in user.preferred_categories:
            fav_cats[cat] += 1
    
    top_cats = sorted(fav_cats.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'\n🏷️ Most preferred categories:')
    for cat, count in top_cats:
        print(f'  • {cat}: {count} users')
    
    print(f'\n💡 Next steps:')
    print(f'  1. Run: python scripts/phase2/build_interaction_matrix.py')
    print(f'  2. Run: python scripts/phase2/train_lightfm.py')
    print(f'  3. Restart your server')
    print(f'  4. Test: curl "http://localhost:8000/api/feed/sim_user_00?mode=fun&count=10"')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    asyncio.run(simulate())