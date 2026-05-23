"""
CLIP multimodal model - text and image embeddings at 512-dim.
Model: openai/clip-vit-base-patch32
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def _load() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def embed_text_clip(text: str) -> list[float]:
    """Embed a text string using CLIP's text encoder -> 512-dim normalised vector."""
    model, processor = _load()
    inputs = processor(
        text=[text[:77]],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()


def embed_image_clip(image: Image.Image) -> list[float]:
    """Embed a PIL image using CLIP's image encoder -> 512-dim normalised vector."""
    model, processor = _load()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()


def embed_post_multimodal(
    text: str,
    image: Image.Image | None = None,
    alpha: float = 0.5,
) -> list[float]:
    """
    Fuse text + image embeddings with a weighted average.
    alpha=1.0 -> text only, alpha=0.0 -> image only.
    Returns normalised 512-dim vector.
    """
    text_emb = np.array(embed_text_clip(text))
    if image is None:
        return text_emb.tolist()
    img_emb = np.array(embed_image_clip(image))
    combined = alpha * text_emb + (1.0 - alpha) * img_emb
    norm = np.linalg.norm(combined)
    return (combined / norm if norm > 0 else combined).tolist()


def classify_zero_shot(
    image: Image.Image,
    labels: list[str],
) -> dict[str, float]:
    """Zero-shot image classification via CLIP similarity scores."""
    model, processor = _load()
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits_per_image.softmax(dim=1)[0]
    return {label: round(float(s), 4) for label, s in zip(labels, logits)}


# -- Quick smoke test -------------------------------------------------
if __name__ == "__main__":
    emb = embed_text_clip("Python machine learning tutorial")
    norm = sum(x ** 2 for x in emb) ** 0.5
    print(f"Text embedding dim={len(emb)}, norm={norm:.4f}")
    assert len(emb) == 512, f"Expected 512, got {len(emb)}"
    assert abs(norm - 1.0) < 0.01, "Vector not normalised"
    print("clip_model.py - PASS")
