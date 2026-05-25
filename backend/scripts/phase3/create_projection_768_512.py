import numpy as np
from pathlib import Path

DIM_IN = 768
DIM_OUT = 512

# Créer une matrice de projection aléatoire
np.random.seed(42)
proj = np.random.randn(DIM_IN, DIM_OUT).astype('float32')

# Normaliser les colonnes
proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)

# Sauvegarder
output_path = Path("data/processed/proj_768_512.npy")
np.save(output_path, proj)

print(f"✅ Matrice de projection créée: {proj.shape}")
print(f"   Sauvegardée: {output_path}")