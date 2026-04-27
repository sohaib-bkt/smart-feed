import pandas as pd

# Load from local CSV
df = pd.read_csv("data/raw/jigsaw_train_hate_annotationprob.csv")


cols = ['text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[cols].dropna()

print(f"Loaded: {len(df)} comments")
print(f"Toxic rate: {df['toxic'].mean():.1%}")

df.to_parquet("data/raw/jigsaw.parquet", index=False)