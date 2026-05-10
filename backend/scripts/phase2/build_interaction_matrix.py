import numpy as np
import pandas as pd
import asyncio
from scipy.sparse import csr_matrix, save_npz
import sys
sys.path.insert(0, '.')

from app.db.firebase import get_all_interactions

ACTION_WEIGHTS = {
    'like':       1.0,
    'watch_full': 0.8,
    'watch_50':   0.5,
    'watch_25':   0.3,
    'comment':    0.7,
    'share':      0.9,
    'skip':      -0.3,
    'hide':      -0.8,
}

async def build():
    print('Chargement des interactions Firebase...')
    interactions = await get_all_interactions()
    df = pd.DataFrame(interactions)

    if df.empty:
        print('Aucune interaction trouvée.')
        print('Simuler des données avec scripts/phase2/simulate_interactions.py')
        return

    print(f'Interactions chargées : {len(df)}')

    users    = df['user_id'].unique()
    articles = df['post_id'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    art2idx  = {a: i for i, a in enumerate(articles)}

    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        w = ACTION_WEIGHTS.get(row.get('action',''), 0)
        if w == 0:
            continue
        rows.append(user2idx[row['user_id']])
        cols.append(art2idx[row['post_id']])
        data.append(w)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(articles))
    )
    print(f'Matrice : {matrix.shape}  —  {matrix.nnz} interactions non nulles')
    print(f'Densité : {matrix.nnz / (matrix.shape[0]*matrix.shape[1]):.4%}')

    save_npz('data/processed/interaction_matrix.npz', matrix)
    np.save('data/processed/user_ids.npy',    np.array(users,    dtype=str))
    np.save('data/processed/article_ids.npy', np.array(articles, dtype=str))

    import json
    with open('data/processed/user2idx.json', 'w') as f:
        json.dump({str(k):v for k,v in user2idx.items()}, f)
    with open('data/processed/art2idx.json', 'w') as f:
        json.dump({str(k):v for k,v in art2idx.items()}, f)

    print('Matrice sauvegardée : data/processed/interaction_matrix.npz')

if __name__ == '__main__':
    asyncio.run(build())