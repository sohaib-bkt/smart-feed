import pandas as pd

df = pd.read_csv("data/raw/sentimentdataset.csv")
# Normaliser les colonnes
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
print(df.columns.tolist())
print(df.head(2))

df.to_parquet("data/raw/social_media.parquet", index=False)