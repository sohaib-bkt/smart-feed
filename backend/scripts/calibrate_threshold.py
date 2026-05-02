import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.models.toxicity import score_toxicity

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
SAMPLE_SIZE = 2000


def main() -> None:
    df = pd.read_parquet("data/raw/jigsaw.parquet")
    if df.empty:
        print("No data found in data/raw/jigsaw.parquet")
        return

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    text_col = "comment_text" if "comment_text" in sample.columns else "text"
    texts = sample[text_col].fillna("").astype(str)
    y_true = sample["toxic"].astype(int)

    scores = texts.apply(lambda text: score_toxicity(text)["toxicity_score"])

    best_threshold = None
    best_f1 = -1.0
    for threshold in THRESHOLDS:
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"{threshold:.2f} -> F1: {f1:.4f}")
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1

    print(f"Best threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    if best_f1 < 0.80:
        print("Warning: F1 below 0.80; consider revisiting data or threshold.")
    print("Remember to update TOXICITY_THRESHOLD in .env")


if __name__ == "__main__":
    main()
