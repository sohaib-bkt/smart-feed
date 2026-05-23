"""
Build CLIP embeddings for MS-COCO (first 10 000 images).
Output:
  data/processed/coco_embeddings.npy   shape (10000, 512)
  data/processed/coco_meta.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.models.multimodal.clip_model import embed_post_multimodal

MAX_IMAGES = 10_000


def main() -> None:
    from datasets import load_dataset

    print(f"Loading MS-COCO (streaming, first {MAX_IMAGES} images)...")
    ds = load_dataset(
        "HuggingFaceM4/COCO",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    records: list[dict] = []
    embeddings: list[list[float]] = []

    for i, item in enumerate(ds):
        if i >= MAX_IMAGES:
            break

        caption: str = item["sentences"]["raw"][0]
        image = item["image"]  # PIL.Image

        emb = embed_post_multimodal(caption, image, alpha=0.5)

        records.append(
            {
                "text": caption,
                "category": "visual",
                "source": "coco",
                "modal": "image",
                "cocoid": item.get("cocoid", i),
                "date": None,
            }
        )
        embeddings.append(emb)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{MAX_IMAGES} images processed...")

    emb_arr = np.array(embeddings, dtype="float32")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "coco_embeddings.npy", emb_arr)
    pd.DataFrame(records).to_parquet(out_dir / "coco_meta.parquet", index=False)

    print(f"Done. Shape: {emb_arr.shape}")
    print("Saved: data/processed/coco_embeddings.npy")
    print("Saved: data/processed/coco_meta.parquet")


if __name__ == "__main__":
    main()
