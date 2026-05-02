from functools import lru_cache
from typing import Dict

from detoxify import Detoxify

WEIGHTS: Dict[str, float] = {
    "toxicity": 0.40,
    "severe_toxicity": 0.30,
    "obscene": 0.15,
    "threat": 0.10,
    "insult": 0.04,
    "identity_attack": 0.01,
}


@lru_cache(maxsize=1)
def _get_model() -> Detoxify:
    return Detoxify("original")


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > 512:
        text = text[:512]
    return text


def score_toxicity(text: str) -> dict:
    clean_text = _normalize_text(text)
    scores = _get_model().predict(clean_text)

    detail: Dict[str, float] = {}
    weighted = 0.0
    for key, weight in WEIGHTS.items():
        value = float(scores.get(key, 0.0))
        detail[key] = round(value, 4)
        weighted += value * weight

    return {
        "toxicity_score": round(weighted, 4),
        "toxicity_detail": detail,
    }


def is_toxic(text: str, threshold: float = 0.3) -> bool:
    return bool(score_toxicity(text)["toxicity_score"] > threshold)


def _demo() -> None:
    samples = [
        "Hello world, have a nice day!",
        "I hate you so much, you are worthless!",
    ]
    for sample in samples:
        result = score_toxicity(sample)
        print(f"Text: {sample}")
        print(f"Score: {result['toxicity_score']}")
        print(f"Detail: {result['toxicity_detail']}")
        print("-" * 40)


if __name__ == "__main__":
    _demo()
    
