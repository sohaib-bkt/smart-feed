from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.models.embeddings import embed_batch

df = pd.read_parquet("data/raw/huffpost.parquet")
texts = df["text"].tolist()

print(f"Embedding {len(texts)} posts...")
embeddings = embed_batch(texts)   # shape: [N, 384]

np.save("data/processed/huffpost_embeddings.npy", embeddings)
df.to_parquet("data/processed/huffpost_with_meta.parquet", index=False)
print(f"Done. Shape: {embeddings.shape}")