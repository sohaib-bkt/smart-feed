import json
from pathlib import Path

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_id2label(model_dir: Path, model: AutoModelForSequenceClassification) -> dict:
    label_map_path = model_dir / "label_mapping.json"
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        return {int(k): v for k, v in mapping.get("id2label", {}).items()}

    # Fallback to the model config if a separate mapping is missing.
    config_map = getattr(model.config, "id2label", {})
    return {int(k): v for k, v in config_map.items()}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_dir = repo_root / "models" / "distilbert_huffpost_final"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    id2label = load_id2label(model_dir, model)

    texts = [
        "Just booked a weekend trip to Lisbon and need food recommendations.",
        "Homemade pasta night with fresh basil and garlic was a win.",
        "Our kid starts kindergarten next week and we are excited and nervous.",
        "The new skincare routine finally cleared up my dry patches.",
        "Company earnings beat expectations and the stock jumped this morning.",
        "The election debate last night felt chaotic and unproductive.",
        "Trying to build a healthier sleep routine and cut screen time.",
        "Weekend hike with friends was refreshing and low key.",
        "Just watched a hilarious stand-up special and laughed nonstop.",
        "Looking for tips on keeping a travel budget under control.",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)

    for text, prob in zip(texts, probs):
        topk = torch.topk(prob, k=min(3, prob.numel()))
        print("\nText:", text)
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            label = id2label.get(idx, f"LABEL_{idx}")
            print(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    main()
