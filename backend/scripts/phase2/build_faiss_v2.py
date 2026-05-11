import faiss
import numpy as np
import pandas as pd
import json


def main() -> None:
    # 1. Load HuffPost
    hp_emb = np.load("data/processed/huffpost_embeddings.npy").astype("float32")
    hp_meta = pd.read_parquet("data/processed/huffpost_with_meta.parquet")
    hp_meta["source"] = "huffpost"

    # 2. Load Reddit
    rd_emb = np.load("data/processed/reddit_embeddings.npy").astype("float32")
    rd_meta = pd.read_parquet("data/processed/reddit_meta.parquet")
    rd_meta["source"] = "reddit"

    # 3. Align columns (keep text, category, source; fill missing with defaults)
    for df in [hp_meta, rd_meta]:
        if "text" not in df.columns:
            df["text"] = ""
        if "category" not in df.columns:
            df["category"] = "UNKNOWN"
        if "date" not in df.columns:
            df["date"] = None

    combined_meta = pd.concat(
        [
            hp_meta[["text", "category", "source", "date"]],
            rd_meta[["text", "category", "source", "date"]],
        ],
        ignore_index=True,
    )
    combined_emb = np.vstack([hp_emb, rd_emb])  # shape (N_total, 384)

    # 4. Build IndexIVFFlat
    dim = 384
    nlist = 100
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(combined_emb)
    index.add(combined_emb)
    index.nprobe = 10

    # 5. Save
    faiss.write_index(index, "data/processed/faiss_index_v2.bin")
    combined_meta.to_parquet("data/processed/combined_meta.parquet", index=False)

    manifest = {
        "n_huffpost": len(hp_emb),
        "n_reddit": len(rd_emb),
        "n_total": index.ntotal,
        "dim": dim,
        "nlist": nlist,
        "nprobe": 10,
    }
    with open("data/processed/faiss_v2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"V2 index built: {index.ntotal} vectors (HP={len(hp_emb)}, Reddit={len(rd_emb)})"
    )
    # Quick self-search test
    sims, ids = index.search(combined_emb[0:1], k=5)
    print("Self-search top5 ids:", ids[0], "sims:", sims[0].round(4))


if __name__ == "__main__":
    main()
