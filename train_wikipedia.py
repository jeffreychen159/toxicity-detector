import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import *
from sklearn.model_selection import train_test_split

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


if __name__ == '__main__': 
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
    training(train_exs, dev_exs)
