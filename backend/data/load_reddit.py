from datasets import load_dataset
import pandas as pd


def _resolve_score_column(df: pd.DataFrame) -> str | None:
    for name in ["score", "upvotes", "ups", "upvote_ratio"]:
        if name in df.columns:
            return name
    return None


def main() -> None:
    dataset = load_dataset(
        "reddit_tifu",
        "short",
        split="train",
        trust_remote_code=True,
    )
    df = dataset.to_pandas()

    if "title" not in df.columns:
        raise ValueError("Missing required column: title")

    if "tldr" not in df.columns:
        df["tldr"] = ""

    df["text"] = (df["title"].fillna("") + " " + df["tldr"].fillna("")).str.strip()
    if "subreddit" in df.columns:
        df["category"] = df["subreddit"]
    elif "subreddit_name_prefixed" in df.columns:
        df["category"] = df["subreddit_name_prefixed"]
    else:
        df["category"] = "tifu"

    score_col = _resolve_score_column(df)
    if score_col is None:
        df["score"] = 0
    else:
        df["score"] = df[score_col]

    df = df[df["text"].str.len() >= 20].copy()
    out_df = df[["text", "category", "score"]]

    out_df.to_parquet("data/raw/reddit.parquet", index=False)
    print(f"Loaded {len(out_df)} rows, {out_df['category'].nunique()} subreddits")


if __name__ == "__main__":
    main()
