from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.models.embeddings import embed_batch


def main() -> None:
    df = pd.read_parquet("data/raw/reddit.parquet")
    texts = df["text"].tolist()

    print(f"Embedding {len(texts)} Reddit posts...")
    embeddings = embed_batch(texts).astype("float32")

    np.save("data/processed/reddit_embeddings.npy", embeddings)
    df.to_parquet("data/processed/reddit_meta.parquet", index=False)
    print(f"Done. Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
