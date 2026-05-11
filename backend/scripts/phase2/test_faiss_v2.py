import faiss
import numpy as np
import pandas as pd


def main() -> None:
    try:
        index = faiss.read_index("data/processed/faiss_index_v2.bin")
        meta = pd.read_parquet("data/processed/combined_meta.parquet")
        index.nprobe = 10
        hp_emb = np.load("data/processed/huffpost_embeddings.npy").astype("float32")
        rd_emb = np.load("data/processed/reddit_embeddings.npy").astype("float32")

        if len(hp_emb) == 0:
            raise ValueError("huffpost_embeddings.npy is empty")

        hp_count = len(hp_emb)
        query = hp_emb[0:1]
        sims, ids = index.search(query, k=20)
        rd_query = rd_emb[0:1]
        _, rd_ids = index.search(rd_query, k=20)

        if index.ntotal <= hp_count:
            raise AssertionError(
                f"Index total ({index.ntotal}) not greater than HuffPost count ({hp_count})"
            )

        valid_ids = [i for i in ids[0] if i >= 0 and i < len(meta)]
        valid_rd_ids = [i for i in rd_ids[0] if i >= 0 and i < len(meta)]
        combined_ids = valid_ids + valid_rd_ids
        sources = set(meta.iloc[combined_ids]["source"].tolist())

        if "reddit" not in sources or "huffpost" not in sources:
            raise AssertionError(
                f"Missing sources in results: {sorted(sources)}"
            )

        print("Top 5 results:")
        for sim, idx in zip(sims[0][:5], ids[0][:5]):
            if idx < 0 or idx >= len(meta):
                continue
            row = meta.iloc[idx]
            text = str(row.get("text", ""))[:80]
            category = row.get("category", "")
            source = row.get("source", "")
            print(f"- {text} | {category} | {source} | {float(sim):.4f}")

        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")


if __name__ == "__main__":
    main()
