import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import *
from sklearn.model_selection import train_test_split
from bert import *

def prepare_labels(df):
    # Typical jigsaw labels
    label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Expected label column '{c}' in the CSV")
    y = df[label_cols].values.astype(np.float32)
    return y, label_cols

def prepare_X(df):
    # Placeholder - feature extraction is handled in the main routine using TfidfSVDTransformer
    texts = df['comment_text'].fillna("").values
    return texts

def prepare_MLP(): 
    # Reading data
    train_csv = pd.read_csv("train.csv")

    # Preparing data
    y, label_cols = prepare_labels(train_csv)

    texts = prepare_X(train_csv)

    transformer = TfidfSVDTransformer()
    X = transformer.fit_transform(texts)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # PLAY AROUND WITH THESE VALUES
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_exs = (X_train, y_train)
    dev_exs = (X_val, y_val)

    # Train Model
    train_MLP(train_exs, dev_exs)
    
def prepare_bertweet_dropped():
    # Load CSV
    df = pd.read_csv("train.csv").sample(frac=0.1, random_state=42) 

    # Prepare labels
    y, label_cols = prepare_labels(df)
    texts = prepare_X(df)

    nontoxic_mask = (y.sum(axis=1) == 0)
    toxic_mask = ~nontoxic_mask

    nontoxic_indices = np.where(nontoxic_mask)[0]
    toxic_indices = np.where(toxic_mask)[0]

    np.random.seed(42)
    num_to_drop = int(0.9 * len(nontoxic_indices))

    drop_indices = np.random.choice(
        nontoxic_indices,
        size=num_to_drop,
        replace=False
    )

    # Mask to keep everything except the dropped rows
    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[drop_indices] = False

    texts = texts[keep_mask]
    y = y[keep_mask]

    print(f"Original dataset size: {len(df)}")
    print(f"Non-toxic samples: {len(nontoxic_indices)}")
    print(f"Dropped non-toxic samples: {num_to_drop}")
    print(f"Final dataset size: {len(texts)}")

    train_texts, val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=42, shuffle=True
    )

    train_bertweet(train_texts, y_train, val_texts, y_val)

def prepare_bertweet(): 
    # Load CSV
    df = pd.read_csv("train.csv")

    # Prepare data
    y, label_cols = prepare_labels(df)
    texts = prepare_X(df)

    # Train/Val Split
    train_texts, val_texts, y_train, y_val = train_test_split(texts, y, test_size=0.2, random_state=42)

    # Train BERTweet
    train_bertweet(train_texts, y_train, val_texts, y_val)

if __name__ == '__main__': 
    # prepare_MLP()
    prepare_bertweet_dropped()