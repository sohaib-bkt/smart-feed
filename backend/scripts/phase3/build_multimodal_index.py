# scripts/phase3/build_multimodal_index.py (corrigé avec projection vidéo)

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
DIM_VIDEO_IN = 500  # Dimensions de vos embeddings vidéo


# -- Helpers ----------------------------------------------------------

def load_or_create_projection(path: Path) -> np.ndarray:
    """Load a saved 384->512 projection matrix or create a new orthonormal one."""
    if path.exists():
        proj = np.load(path).astype("float32")
        print(f"  Loaded projection matrix from {path}")
    else:
        rng = np.random.default_rng(seed=42)
        raw = rng.standard_normal((DIM_IN, DIM_OUT)).astype("float32")
        proj, _ = np.linalg.qr(raw)
        proj = np.hstack([proj, raw[:, :DIM_OUT - DIM_IN]]).astype("float32")
        proj /= np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1e-8)
        np.save(path, proj)
        print(f"  Created new projection matrix -> {path}")
    return proj


def load_or_create_video_projection(path: Path, dim_in: int, dim_out: int) -> np.ndarray:
    """Load or create a projection matrix for video embeddings (500 -> 512)."""
    if path.exists():
        proj = np.load(path).astype("float32")
        print(f"  Loaded video projection matrix from {path}")
    else:
        rng = np.random.default_rng(seed=42)
        raw = rng.standard_normal((dim_in, dim_out)).astype("float32")
        # Normaliser les colonnes
        proj = raw / np.linalg.norm(raw, axis=0, keepdims=True).clip(min=1e-8)
        np.save(path, proj)
        print(f"  Created new video projection matrix -> {path}")
    return proj


def project_and_normalise(emb_384: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Project (N, 384) -> (N, 512) and L2-normalise each row."""
    emb_512 = (emb_384.astype("float32") @ proj).astype("float32")
    norms = np.linalg.norm(emb_512, axis=1, keepdims=True).clip(min=1e-8)
    return emb_512 / norms


def project_video_and_normalise(emb_video: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """Project video embeddings (N, 500) -> (N, 512) and normalise."""
    emb_512 = (emb_video.astype("float32") @ proj).astype("float32")
    norms = np.linalg.norm(emb_512, axis=1, keepdims=True).clip(min=1e-8)
    return emb_512 / norms


# -- Main -------------------------------------------------------------

def main() -> None:
    proj_path = OUT_DIR / "proj_384_512.npy"
    proj = load_or_create_projection(proj_path)
    
    video_proj_path = OUT_DIR / "proj_video_500_512.npy"
    video_proj = load_or_create_video_projection(video_proj_path, DIM_VIDEO_IN, DIM_OUT)

    # 1. HuffPost text (384 -> 512)
    print("\n[1/5] Loading HuffPost embeddings...")
    hp_emb = np.load(OUT_DIR / "huffpost_embeddings.npy")
    hp_meta = pd.read_parquet(OUT_DIR / "huffpost_with_meta.parquet")
    hp_meta = hp_meta[["text", "category"]].copy()
    hp_meta["source"] = "huffpost"
    hp_meta["modal"] = "text"
    hp_meta["date"] = None
    hp_512 = project_and_normalise(hp_emb, proj)
    print(f"  HuffPost: {hp_512.shape}")

    # 2. Reddit text (384 -> 512)
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
        
        available_cols = coco_meta.columns.tolist()
        print(f"  COCO columns: {available_cols}")
        
        # Construire les colonnes nécessaires
        coco_meta_clean = pd.DataFrame()
        coco_meta_clean["text"] = coco_meta.get("text", coco_meta.get("caption", ""))
        coco_meta_clean["category"] = coco_meta.get("category", "image")
        coco_meta_clean["source"] = "coco"
        coco_meta_clean["modal"] = "image"
        coco_meta_clean["date"] = None
        
        print(f"  COCO: {coco_emb.shape}")
    else:
        print("\n[3/5] COCO embeddings not found - skipping image source.")
        coco_emb = np.empty((0, DIM_OUT), dtype="float32")
        coco_meta_clean = pd.DataFrame(columns=["text", "category", "source", "modal", "date"])

    # 4. Video embeddings (500 -> 512 via projection)
    video_path = OUT_DIR / "videomae_embeddings.npy"
    if video_path.exists():
        print("\n[4/5] Loading video embeddings...")
        vid_emb_raw = np.load(video_path).astype("float32")
        print(f"  Video raw shape: {vid_emb_raw.shape}")
        
        # Projeter de 500d à 512d
        vid_emb = project_video_and_normalise(vid_emb_raw, video_proj)
        print(f"  Video projected shape: {vid_emb.shape}")
        
        vid_meta = pd.read_parquet(OUT_DIR / "videomae_meta.parquet")
        print(f"  Video columns: {vid_meta.columns.tolist()}")
        
        # Adapter les métadonnées
        vid_meta_clean = pd.DataFrame()
        
        if 'label_name' in vid_meta.columns:
            vid_meta_clean["text"] = vid_meta["label_name"]
        elif 'video_path' in vid_meta.columns:
            vid_meta_clean["text"] = vid_meta["video_path"].apply(lambda x: str(x).split("/")[-1])
        else:
            vid_meta_clean["text"] = "video content"
        
        if 'label_name' in vid_meta.columns:
            vid_meta_clean["category"] = vid_meta["label_name"]
        else:
            vid_meta_clean["category"] = "video"
        
        vid_meta_clean["source"] = "video"
        vid_meta_clean["modal"] = "video"
        vid_meta_clean["date"] = None
        
        print(f"  Video metadata: {vid_meta_clean.shape}")
    else:
        print("\n[4/5] No video embeddings found - adding minimal stub.")
        N_STUB = 10
        rng = np.random.default_rng(42)
        vid_emb = rng.standard_normal((N_STUB, DIM_OUT)).astype("float32")
        norms = np.linalg.norm(vid_emb, axis=1, keepdims=True).clip(min=1e-8)
        vid_emb /= norms
        vid_meta_clean = pd.DataFrame(
            [
                {"text": f"video stub {i}", "category": "video", "source": "stub", "modal": "video", "date": None}
                for i in range(N_STUB)
            ]
        )

    # 5. Combine all sources
    print("\n[5/5] Building FAISS V3 index...")
    all_emb = np.vstack([hp_512, rd_512, coco_emb, vid_emb]).astype("float32")
    all_meta = pd.concat(
        [hp_meta, rd_meta, coco_meta_clean, vid_meta_clean], ignore_index=True
    )

    assert len(all_emb) == len(all_meta), (
        f"Mismatch: {len(all_emb)} embeddings vs {len(all_meta)} metadata rows"
    )

    print(f"  Total vectors: {len(all_emb)}")
    print(f"  Breakdown - HP:{len(hp_512)} Reddit:{len(rd_512)} COCO:{len(coco_emb)} Video:{len(vid_emb)}")

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
        "proj_video_500_512": str(video_proj_path),
    }
    with open(OUT_DIR / "faiss_v3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ V3 index built: {index.ntotal} vectors at {dim}-dim")
    print("   Saved: data/processed/faiss_index_v3.bin")
    print("   Saved: data/processed/combined_meta_v3.parquet")
    print("   Saved: data/processed/faiss_v3_manifest.json")

    # Quick self-search smoke test
    sims, ids = index.search(all_emb[0:1], k=5)
    print(f"\nSelf-search top5 ids: {ids[0]}  sims: {sims[0].round(4)}")


if __name__ == "__main__":
    main()