import numpy as np
from scipy.sparse import load_npz, csr_matrix
import pickle
import os

# Suppress OpenBLAS warning
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import implicit

os.makedirs('app/models', exist_ok=True)

print('Chargement de la matrice d\'interactions...')
matrix = load_npz('data/processed/interaction_matrix.npz')
print(f'Matrice originale : {matrix.shape}  —  {matrix.nnz} interactions')
print(f'Valeurs min/max: {matrix.data.min():.2f} / {matrix.data.max():.2f}')
print(f'Format: {matrix.shape[0]} utilisateurs, {matrix.shape[1]} articles')

if matrix.shape[0] < 5 or matrix.shape[1] < 5:
    raise ValueError('Pas assez d\'interactions. Lancer simulate_interactions.py')

# IMPORTANT: For Implicit ALS, we need (items x users) or (users x items)?
# Let's try both and see which works

# Option: Keep as (users x items) - standard for recommendation
print('\nOption: Utilisation de la matrice (utilisateurs x articles)...')
confidence_matrix = matrix.astype(np.float32)

# Transform negative values to positives
print('Conversion des valeurs négatives en positives...')
data = confidence_matrix.data.copy()
for i in range(len(data)):
    if data[i] < 0:
        data[i] = 0.05 + (data[i] + 1) * 0.05
    data[i] = np.clip(data[i], 0.01, 2.0)
confidence_matrix.data = data

print(f'Matrice finale: {confidence_matrix.shape}, valeurs [{confidence_matrix.data.min():.3f}, {confidence_matrix.data.max():.3f}]')
print(f'Interactions non-nulles: {confidence_matrix.nnz}')

# ALS Model
model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=50,
    num_threads=4,
    random_state=42
)

print('\nEntraînement Implicit ALS...')
model.fit(confidence_matrix, show_progress=True)

# Check dimensions
print(f'\nDimensions après entraînement:')
print(f'  - user_factors shape: {model.user_factors.shape}')
print(f'  - item_factors shape: {model.item_factors.shape}')

# Verify model learned something
print('\nVérification du modèle...')
test_predictions = []

# Use the correct dimensions - user_factors[user_idx] for users
if model.user_factors.shape[0] >= 5:
    for user in range(min(5, model.user_factors.shape[0])):
        # Test with first 5 items that exist
        test_items = np.array([i for i in range(min(5, model.item_factors.shape[0]))])
        if len(test_items) > 0:
            # Calculate scores via dot product
            user_factors = model.user_factors[user]
            item_factors = model.item_factors[test_items]
            scores = np.dot(item_factors, user_factors)
            test_predictions.extend(scores)
            print(f'  User {user}: scores = {[f"{s:.4f}" for s in scores]}')
else:
    print(f'  Warning: user_factors has shape {model.user_factors.shape}, expected at least 5 users')

if test_predictions:
    print(f'\n  Plage des scores: [{np.min(test_predictions):.4f}, {np.max(test_predictions):.4f}]')
    print(f'  Variance des scores: {np.var(test_predictions):.6f}')
    print(f'  Std deviation: {np.std(test_predictions):.6f}')

    if np.std(test_predictions) < 0.1:
        print('  ⚠️ Attention: Faible variance - le modèle n\'a pas bien appris')
    else:
        print('  ✅ Bonne variance - le modèle a appris correctement')

# Test recommendation function
print('\nTest de la fonction recommend...')
n_users_to_test = min(3, model.user_factors.shape[0])
for user in range(n_users_to_test):
    try:
        # For implicit, recommend expects user id and the user items matrix
        recommendations = model.recommend(
            user,
            confidence_matrix[user],
            N=5,
            filter_already_liked_items=True
        )
        print(f'  User {user}: Top 5 items = {recommendations[0].tolist()}')
    except Exception as e:
        print(f'  User {user}: Error - {e}')

# Save model
with open('app/models/implicit_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('\n✅ Modèle sauvegardé : app/models/implicit_model.pkl')

# Save matrix info
with open('app/models/matrix_info.pkl', 'wb') as f:
    pickle.dump({
        'shape': matrix.shape,
        'nnz': matrix.nnz,
        'n_users': int(matrix.shape[0]),
        'n_items': int(matrix.shape[1]),
        'min_confidence': float(confidence_matrix.data.min()),
        'max_confidence': float(confidence_matrix.data.max()),
        'model_type': 'AlternatingLeastSquares',
        'factors': 64
    }, f)

print(f'\n📊 Statistiques finales:')
print(f'  - Utilisateurs: {matrix.shape[0]}')
print(f'  - Articles: {matrix.shape[1]}')
print(f'  - Interactions: {matrix.nnz}')
print(f'  - Embedding dimension: {model.item_factors.shape[1]}')