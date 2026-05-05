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