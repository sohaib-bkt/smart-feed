import os

import pandas as pd
from datasets import load_dataset

RAW_CSV_PATH = "data/raw/jigsaw_train_hate_annotationprob.csv"
PARQUET_PATH = "data/raw/jigsaw.parquet"
HF_COLS = [
	"comment_text",
	"toxic",
	"severe_toxic",
	"obscene",
	"threat",
	"insult",
	"identity_hate",
]
CSV_COLS = [
	"text",
	"toxic",
	"severe_toxic",
	"obscene",
	"threat",
	"insult",
	"identity_hate",
]


def _load_from_csv() -> pd.DataFrame:
	df = pd.read_csv(RAW_CSV_PATH)
	df = df[CSV_COLS].dropna()
	return df.rename(columns={"text": "comment_text"})


def _load_from_hf() -> pd.DataFrame:
	dataset = load_dataset("jigsaw_toxicity_pred", split="train")
	dataset = dataset.select_columns(HF_COLS)
	return dataset.to_pandas().dropna()


def main() -> None:
	if os.path.exists(RAW_CSV_PATH):
		df = _load_from_csv()
		source = RAW_CSV_PATH
	else:
		df = _load_from_hf()
		source = "jigsaw_toxicity_pred"

	print(f"Loaded: {len(df)} comments from {source}")
	print(f"Toxic rate: {df['toxic'].mean():.1%}")

	df.to_parquet(PARQUET_PATH, index=False)


if __name__ == "__main__":
	main()