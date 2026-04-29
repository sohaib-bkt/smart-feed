import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("DistilBERT Fine-tuning on News Category Data")
print("=" * 60)

# Step 1: Load data from local JSON file
print("\n[1/5] Loading dataset from local file...")
data_path = './backend/data/raw/News_Category_Dataset_v2.json'

try:
    # Load JSON file
    articles = []
    with open(data_path, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
    
    df = pd.DataFrame(articles)
    print(f"✓ Loaded from {data_path}")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"✗ File not found: {data_path}")
    raise
except Exception as e:
    print(f"✗ Error loading file: {e}")
    raise

# Prepare data
print(f"  Dataset size: {len(df)} samples")

# Identify text and category columns
if 'headline' in df.columns:
    text_col = 'headline'
else:
    text_col = df.columns[0]

if 'category' in df.columns:
    category_col = 'category'
elif 'Category' in df.columns:
    category_col = 'Category'
else:
    category_col = df.columns[-1]

# Create text column combining headline and description if available
if 'short_description' in df.columns:
    df['text'] = df[text_col].fillna('') + " " + df['short_description'].fillna('')
elif 'description' in df.columns:
    df['text'] = df[text_col].fillna('') + " " + df['description'].fillna('')
else:
    df['text'] = df[text_col].fillna('')

# Use category column
df['category'] = df[category_col]
print(f"  Categories found: {df['category'].nunique()}")

# Clean data
print("\n[2/5] Cleaning data...")
df = df.dropna(subset=['text', 'category'])
df = df[df['text'].str.len() > 10]  # Remove very short texts
df = df[df['category'].str.len() > 0]  # Remove empty categories

# Limit to top categories for faster training (optional)
top_categories = df['category'].value_counts().head(10).index
df = df[df['category'].isin(top_categories)]

# Sample data for faster training
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)
    print(f"  Sampled to 5000 articles")

print(f"✓ Cleaned dataset: {len(df)} samples")

# Encode labels
print("\n[3/5] Encoding labels...")
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
# id2label: index -> category name (string keys for id)
id2label = {str(i): label for i, label in enumerate(le.classes_)}
# label2id: category name -> index (string keys for label)
label2id = {label: i for i, label in enumerate(le.classes_)}
num_labels = len(le.classes_)
print(f"✓ {num_labels} categories: {list(le.classes_[:5])}{'...' if num_labels > 5 else ''}")

# Step 2: Split data
print("\n[4/5] Splitting data (80/20 train/test)...")
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]
print(f"✓ Train: {len(train_df)} samples")
print(f"✓ Test:  {len(test_df)} samples")

# Step 3: Create Hugging Face datasets and tokenize
print("\n[5/5] Preparing datasets and tokenizing...")
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print(f"✓ Tokenization complete")

# Step 4: Fine-tune the model
print("\n" + "=" * 60)
print("Fine-tuning DistilBERT...")
print("=" * 60)
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir='./models/distilbert_huffpost',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='loss',
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=data_collator,
)

trainer.train()

# Step 5: Save the model
print("\n" + "=" * 60)
print("Saving fine-tuned model...")
model_save_path = './models/distilbert_huffpost_final'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✓ Model saved to: {model_save_path}")

# Save label encoding for later use
import json
with open(os.path.join(model_save_path, 'label_mapping.json'), 'w') as f:
    json.dump({'id2label': {str(k): v for k, v in id2label.items()},
               'label2id': {k: v for k, v in label2id.items()}}, f)
print(f"✓ Label mapping saved")

print("\n" + "=" * 60)
print("Fine-tuning completed successfully!")
print("=" * 60)
