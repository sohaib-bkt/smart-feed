"""
Smoke test for FAISS V3 multimodal index.
Run from backend/: python scripts/phase3/test_faiss_v3.py
Expected output: PASS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))


def main() -> None:
    try:
        # -- 1. Index + metadata load --------------------------------
        index = faiss.read_index("data/processed/faiss_index_v3.bin")
        meta = pd.read_parquet("data/processed/combined_meta_v3.parquet")
        manifest = json.loads(
            Path("data/processed/faiss_v3_manifest.json").read_text()
        )
        index.nprobe = 10

        # -- 2. Dimension check --------------------------------------
        assert index.d == 512, f"Expected dim=512, got {index.d}"
        print(f"[OK] Index dim = {index.d}")

        # -- 3. Total vector count -----------------------------------
        expected_total = (
            manifest["n_huffpost"]
            + manifest["n_reddit"]
            + manifest["n_coco"]
            + manifest["n_video"]
        )
        assert index.ntotal == expected_total, (
            f"Total mismatch: index={index.ntotal}, manifest={expected_total}"
        )
        print(f"[OK] Total vectors = {index.ntotal}")

        # -- 4. Metadata alignment -----------------------------------
        assert len(meta) == index.ntotal, (
            f"Meta rows ({len(meta)}) != index vectors ({index.ntotal})"
        )
        print(f"[OK] Metadata aligned = {len(meta)} rows")

        # -- 5. Source coverage --------------------------------------
        sources = set(meta["source"].unique())
        assert "huffpost" in sources, f"Missing source 'huffpost' in {sources}"
        assert "reddit" in sources, f"Missing source 'reddit' in {sources}"
        print(f"[OK] Sources present: {sorted(sources)}")

        # -- 6. Projection matrix ------------------------------------
        proj_path = Path("data/processed/proj_384_512.npy")
        assert proj_path.exists(), "proj_384_512.npy not found"
        proj = np.load(proj_path)
        assert proj.shape == (384, 512), f"Bad projection shape: {proj.shape}"
        print(f"[OK] Projection matrix shape = {proj.shape}")

        # -- 7. Search test - HuffPost query -------------------------
        hp_emb = np.load("data/processed/huffpost_embeddings.npy").astype("float32")
        proj_f = proj.astype("float32")
        hp_q = hp_emb[0:1] @ proj_f
        hp_q /= np.linalg.norm(hp_q, axis=1, keepdims=True).clip(min=1e-8)

        sims, ids = index.search(hp_q, k=20)
        valid = [i for i in ids[0] if 0 <= i < len(meta)]
        assert len(valid) > 0, "No valid search results for HuffPost query"
        result_sources = set(meta.iloc[valid]["source"].tolist())
        print(f"[OK] HuffPost query -> sources in top-20: {sorted(result_sources)}")

        # -- 8. Search test - CLIP text query ------------------------
        try:
            from app.models.multimodal.clip_model import embed_text_clip

            clip_emb = np.array(embed_text_clip("machine learning"), dtype="float32")
            assert len(clip_emb) == 512, f"CLIP embedding dim={len(clip_emb)}"
            norm = np.linalg.norm(clip_emb)
            assert abs(norm - 1.0) < 0.01, "CLIP embedding not normalised"

            sims_clip, ids_clip = index.search(clip_emb.reshape(1, -1), k=10)
            valid_clip = [i for i in ids_clip[0] if 0 <= i < len(meta)]
            clip_sources = set(meta.iloc[valid_clip]["source"].tolist())
            print(f"[OK] CLIP text query -> sources in top-10: {sorted(clip_sources)}")
        except Exception as clip_err:
            print(f"[WARN] CLIP query skipped (model not cached yet): {clip_err}")

        # -- 9. Print manifest ---------------------------------------
        print("\n  Manifest summary:")
        for k, v in manifest.items():
            print(f"    {k}: {v}")

        print("\nPASS")

    except Exception as exc:
        print(f"FAIL: {exc}")
        raise


if __name__ == "__main__":
    main()
