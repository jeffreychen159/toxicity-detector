import pandas as pd
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import *
from sklearn.model_selection import train_test_split
from bert import *


def prepare_labels():
    df = pd.read_json(
        "hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl",
        lines=True
    )

    # 8 regression labels
    label_cols = [
        "toxicity", "profanity", "sexually_explicit", "flirtation",
        "identity_attack", "threat", "insult", "severe_toxicity"
    ]

    rows = []

    for _, row in df.iterrows():

        # PROMPT
        p = row["prompt"]
        rows.append({
            "text": p["text"],
            **{label: p[label] for label in label_cols}
        })

        # CONTINUATION
        c = row["continuation"]
        if isinstance(c, dict) and "text" in c:
            rows.append({
                "text": c["text"],
                **{label: c[label] for label in label_cols}
            })

    clean_df = pd.DataFrame(rows)

    texts = clean_df["text"].fillna("").values
    labels = clean_df[label_cols].values.astype(np.float32)

    train_texts, val_texts, y_train, y_val = train_test_split(
        texts, 
        labels,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    train_bertweet(train_texts, y_train, val_texts, y_val)

if __name__ == '__main__': 
    prepare_labels()