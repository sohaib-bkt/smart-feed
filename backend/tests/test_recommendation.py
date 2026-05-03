# recommend_similar.py
import faiss
import numpy as np
import pandas as pd

# Load data
df = pd.read_parquet("data/raw/huffpost.parquet")
index = faiss.read_index("data/processed/faiss_index.bin")

def recommend_similar(article_id, k=10):
    """Recommend articles similar to a given article"""
    embedding = np.load("data/processed/huffpost_embeddings.npy")[article_id]
    scores, indices = index.search(np.array([embedding]).astype("float32"), k+1)
    
    # Skip the first result (it's the article itself)
    recommendations = indices[0][1:k+1]
    return recommendations

# Example: Find articles similar to article #0
similar_articles = recommend_similar(0)
print(f"Similar to: {df.iloc[0]['headline']}")
for idx in similar_articles:
    print(f"- {df.iloc[idx]['headline']} ({df.iloc[idx]['category']})")