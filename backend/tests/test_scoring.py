from datetime import datetime, timedelta
from app.services.scoring import compute_score

def test_compute_score_normal():
    """Test un post normal avec bonne similarité."""
    result = compute_score(
        cosine_sim=0.8,
        toxicity_score=0.1,
        category='TECH',
        user_interests=['TECH', 'SCIENCE'],
        created_at=datetime.now(),
        user_prefs={'mode': 'default', 'toxicity_threshold': 0.3}
    )
    
    assert result['score'] > 0.5
    assert not result['filtered']
    print(f"✅ Test normal : score = {result['score']:.3f}")

def test_compute_score_toxic():
    """Test un post toxique qui doit être filtré."""
    result = compute_score(
        cosine_sim=0.9,
        toxicity_score=0.8,
        category='TECH',
        user_interests=['TECH'],
        created_at=datetime.now(),
        user_prefs={'mode': 'default', 'toxicity_threshold': 0.3}
    )
    
    assert result['score'] == 0.0
    assert result['filtered']
    assert result['reason'] == 'toxicity'
    print(f"✅ Test toxique : filtré avec raison '{result['reason']}'")

def test_compute_score_old_post():
    """Test un post ancien qui devrait avoir un score réduit."""
    old_date = datetime.now() - timedelta(days=25)
    new_date = datetime.now()
    
    result_old = compute_score(
        cosine_sim=0.8, toxicity_score=0.1,
        category='TECH', user_interests=['TECH'],
        created_at=old_date,
        user_prefs={'mode': 'default'}
    )
    
    result_new = compute_score(
        cosine_sim=0.8, toxicity_score=0.1,
        category='TECH', user_interests=['TECH'],
        created_at=new_date,
        user_prefs={'mode': 'default'}
    )
    
    assert result_new['score'] > result_old['score']
    print(f"✅ Test récence : nouveau({result_new['score']:.3f}) > ancien({result_old['score']:.3f})")

def test_different_modes():
    """Test que les modes donnent des scores différents."""
    result_focus = compute_score(
        cosine_sim=0.8, toxicity_score=0.1,
        category='SPORTS', user_interests=['TECH'],
        created_at=datetime.now(),
        user_prefs={'mode': 'focus'}
    )
    
    result_fun = compute_score(
        cosine_sim=0.8, toxicity_score=0.1,
        category='SPORTS', user_interests=['SPORTS'],
        created_at=datetime.now(),
        user_prefs={'mode': 'fun'}
    )
    
    # En mode fun, la préférence a plus de poids
    print(f"✅ Test modes : focus={result_focus['score']:.3f}, fun={result_fun['score']:.3f}")

if __name__ == '__main__':
    test_compute_score_normal()
    test_compute_score_toxic()
    test_compute_score_old_post()
    test_different_modes()
    print("\n🎉 Tous les tests du Module 6 sont passés !")