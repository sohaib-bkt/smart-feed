from datasets import load_dataset
import pandas as pd

# Load the test split specifically
ds = load_dataset("khalidalt/HuffPost", split="test", trust_remote_code=True)
df = ds.to_pandas()

# Garder seulement les colonnes utiles
df = df[["headline","short_description","category","date"]].dropna()
df["text"] = df["headline"] + " " + df["short_description"]

print(f"Loaded: {len(df)} articles, {df['category'].nunique()} categories")
print(df["category"].value_counts().head(10))

df.to_parquet("data/raw/huffpost.parquet", index=False)