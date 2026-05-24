"""
Build FAISS V3 - unified multimodal index at 512 dimensions.

Sources:
  - HuffPost text   (384 -> 512 via linear projection)
  - Reddit text     (384 -> 512 via same projection)
  - COCO images     (512 - already correct, from CLIP)
  - Video stub      (512 - zero stub if no real video embeddings exist)

Output:
  data/processed/faiss_index_v3.bin
  data/processed/combined_meta_v3.parquet
  data/processed/proj_384_512.npy   (projection matrix, saved for inference)
  data/processed/faiss_v3_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_IN = 384
DIM_OUT = 512


# -- Helpers ----------------------------------------------------------

def load_or_create_projection(path: Path) -> np.ndarray:
    """
    Load a saved 384->512 projection matrix or create a new orthonormal one.
    Re-using the same matrix ensures consistency between build and inference.
    """
    if path.exists():
        proj = np.load(path).astype("float32")
        print(f"  Loaded projection matrix from {path}")
    else:
        rng = np.random.default_rng(seed=42)
        # Random Gaussian -> QR decomposition for near-orthogonal columns
        raw = rng.standard_normal((DIM_IN, DIM_OUT)).astype("float32")
        proj, _ = np.linalg.qr(raw)  # shape (384, 384) - take first 384 cols
        # Pad to (384, 512): duplicate / extend
        proj = np.hstack([proj, raw[:, :DIM_OUT - DIM_IN]]).astype("float32")
        # Column-normalise
        proj /= np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1e-8)
        np.save(path, proj)
        print(f"  Created new projection matrix -> {path}")
    return proj


def project_and_normalise(emb_384: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Project (N, 384) -> (N, 512) and L2-normalise each row."""
    emb_512 = (emb_384.astype("float32") @ proj).astype("float32")
    norms = np.linalg.norm(emb_512, axis=1, keepdims=True).clip(min=1e-8)
    return emb_512 / norms


# -- Main -------------------------------------------------------------

def main() -> None:
    proj_path = OUT_DIR / "proj_384_512.npy"
    proj = load_or_create_projection(proj_path)

    # 1. HuffPost text (384 -> 512)
    print("\n[1/5] Loading HuffPost embeddings...")
    hp_emb = np.load(OUT_DIR / "huffpost_embeddings.npy")
    hp_meta = pd.read_parquet(OUT_DIR / "huffpost_with_meta.parquet")
    hp_meta = hp_meta[["text", "category", "date"]].copy()
    hp_meta["source"] = "huffpost"
    hp_meta["modal"] = "text"
    hp_512 = project_and_normalise(hp_emb, proj)
    print(f"  HuffPost: {hp_512.shape}")

    # 2. Reddit text (384 -> 512) - same projection matrix
    print("\n[2/5] Loading Reddit embeddings...")
    rd_emb = np.load(OUT_DIR / "reddit_embeddings.npy")
    rd_meta = pd.read_parquet(OUT_DIR / "reddit_meta.parquet")
    rd_meta = rd_meta[["text", "category"]].copy()
    rd_meta["source"] = "reddit"
    rd_meta["modal"] = "text"
    rd_meta["date"] = None
    rd_512 = project_and_normalise(rd_emb, proj)
    print(f"  Reddit: {rd_512.shape}")

    # 3. COCO images (already 512-dim from CLIP)
    coco_path = OUT_DIR / "coco_embeddings.npy"
    if coco_path.exists():
        print("\n[3/5] Loading COCO embeddings...")
        coco_emb = np.load(coco_path).astype("float32")
        coco_meta = pd.read_parquet(OUT_DIR / "coco_meta.parquet")
        coco_meta = coco_meta[["text", "category", "source", "modal", "date"]].copy()
        print(f"  COCO: {coco_emb.shape}")
    else:
        print("\n[3/5] COCO embeddings not found - skipping image source.")
        print("  Run scripts/phase3/build_coco_embeddings.py first for full multimodal support.")
        coco_emb = np.empty((0, DIM_OUT), dtype="float32")
        coco_meta = pd.DataFrame(columns=["text", "category", "source", "modal", "date"])

    # 4. Video stub (512-dim) - placeholder until real VideoMAE embeddings exist
    video_path = OUT_DIR / "video_embeddings.npy"
    if video_path.exists():
        print("\n[4/5] Loading video embeddings...")
        vid_emb = np.load(video_path).astype("float32")
        vid_meta = pd.read_parquet(OUT_DIR / "video_meta.parquet")
        vid_meta = vid_meta[["text", "category", "source", "modal", "date"]].copy()
        print(f"  Video: {vid_emb.shape}")
    else:
        print("\n[4/5] No video embeddings found - adding minimal stub for index compatibility.")
        N_STUB = 10
        rng = np.random.default_rng(42)
        vid_emb = rng.standard_normal((N_STUB, DIM_OUT)).astype("float32")
        norms = np.linalg.norm(vid_emb, axis=1, keepdims=True).clip(min=1e-8)
        vid_emb /= norms
        vid_meta = pd.DataFrame(
            [
                {
                    "text": f"video stub {i}",
                    "category": "video",
                    "source": "stub",
                    "modal": "video",
                    "date": None,
                }
                for i in range(N_STUB)
            ]
        )

    # 5. Combine all sources
    print("\n[5/5] Building FAISS V3 index...")
    all_emb = np.vstack([hp_512, rd_512, coco_emb, vid_emb]).astype("float32")
    all_meta = pd.concat(
        [hp_meta, rd_meta, coco_meta, vid_meta], ignore_index=True
    )

    assert len(all_emb) == len(all_meta), (
        f"Mismatch: {len(all_emb)} embeddings vs {len(all_meta)} metadata rows"
    )

    print(f"  Total vectors: {len(all_emb)}")
    print(
        f"  Breakdown - HP:{len(hp_512)} Reddit:{len(rd_512)} COCO:{len(coco_emb)} Video:{len(vid_emb)}"
    )

    dim = DIM_OUT
    nlist = max(100, int(np.sqrt(len(all_emb))))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(all_emb)
    index.add(all_emb)
    index.nprobe = 10

    faiss.write_index(index, str(OUT_DIR / "faiss_index_v3.bin"))
    all_meta.to_parquet(OUT_DIR / "combined_meta_v3.parquet", index=False)

    manifest = {
        "n_huffpost": int(len(hp_512)),
        "n_reddit": int(len(rd_512)),
        "n_coco": int(len(coco_emb)),
        "n_video": int(len(vid_emb)),
        "n_total": int(index.ntotal),
        "dim": dim,
        "nlist": nlist,
        "nprobe": 10,
        "proj_384_512": str(proj_path),
    }
    with open(OUT_DIR / "faiss_v3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nOK - V3 index built: {index.ntotal} vectors at {dim}-dim")
    print("   Saved: data/processed/faiss_index_v3.bin")
    print("   Saved: data/processed/combined_meta_v3.parquet")
    print("   Saved: data/processed/faiss_v3_manifest.json")

    # Quick self-search smoke test
    sims, ids = index.search(all_emb[0:1], k=5)
    print(f"\nSelf-search top5 ids: {ids[0]}  sims: {sims[0].round(4)}")


if __name__ == "__main__":
    main()
