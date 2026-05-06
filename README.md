# Smart Feed

Team sync after LFS history rewrite
- Install Git LFS once: `git lfs install`
- Safest option: re-clone the repo, then run `git lfs pull`
- If you have no local changes, you can reset instead:
	1) `git fetch --all`
	2) `git checkout main`
	3) `git reset --hard origin/main`
	4) `git lfs pull`

Toxicity module (Phase 1)
- Load Jigsaw data (uses local CSV if present):
	`cd backend`
	`python data/load_jigsaw.py`
- Calibrate threshold (uses 2k sample, prints best F1):
	`python scripts/calibrate_threshold.py`
- Update `.env` with the recommended threshold (current best: 0.20)
- Quick smoke test:
	`python -m app.models.toxicity`
- Run unit tests:
	`pytest tests/test_toxicity.py -v`

Firebase

Point d'entrée de l'API Smart Feed.

- _Lancer_ :   `uvicorn app.main:app --reload --port 8000`
- _Swagger_ :  `http://localhost:8000/docs`

Pour le test avec curl dans le terminal :
```
# 1. Créer un autre utilisateur
curl "http://localhost:8000/api/feed/bob?limit=5"

# 2. Ajouter une interaction
curl -X POST "http://localhost:8000/api/interact" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "bob",
    "post_id": "post_99",
    "post_text": "Les bienfaits du sport sur la santé",
    "action": "like",
    "watch_time": 0.0
  }'

# 3. Voir les préférences
curl "http://localhost:8000/api/preferences/bob"

# 4. Changer le mode
curl -X POST "http://localhost:8000/api/preferences/bob/mode/learning"

# 5. Revérifier le feed après interactions
curl "http://localhost:8000/api/feed/bob?limit=10"
```
