import faiss
import numpy as np

embeddings = np.load("data/processed/huffpost_embeddings.npy")
embeddings = embeddings.astype("float32")

dim = embeddings.shape[1]   # 384
index = faiss.IndexFlatIP(dim)   # Inner Product = cosine (si normalisé)
index.add(embeddings)

faiss.write_index(index, "data/processed/faiss_index.bin")
print(f"Index built: {index.ntotal} vectors, dim={dim}")
