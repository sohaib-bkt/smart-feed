Smart Feed

Team sync after LFS history rewrite
- Install Git LFS once: `git lfs install`
- Safest option: re-clone the repo, then run `git lfs pull`
- If you have no local changes, you can reset instead:
	1) `git fetch --all`
	2) `git checkout main`
	3) `git reset --hard origin/main`
	4) `git lfs pull`