"""Evaluate saved MLP (`mlp_model.pth`) on a 10% sample of `train.csv`.
Saves metrics to `mlp_train_sample_eval.json`.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

from model import MLP, TfidfSVDTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

MLP_PTH = 'mlp_model.pth'
TRAIN_CSV = 'train.csv'
SAMPLE_FRAC = 0.10
RANDOM_STATE = 42
LABEL_COLS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

if not os.path.exists(MLP_PTH):
    raise FileNotFoundError(f"{MLP_PTH} not found in project root")
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"{TRAIN_CSV} not found in project root")

print("Loading train.csv and sampling...")
df = pd.read_csv(TRAIN_CSV)
if not all(c in df.columns for c in LABEL_COLS):
    raise ValueError("Missing expected label columns in train.csv")

df_sample = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
texts = df_sample['comment_text'].fillna('').astype(str).tolist()
labels = df_sample[LABEL_COLS].values

# Split train/val
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=RANDOM_STATE, stratify=(labels.sum(axis=1) > 0)
)

print(f"Sampled {len(df_sample)} rows; validation set size {len(val_texts)}")

# Fit TF-IDF + SVD transformer on train_texts
transformer = TfidfSVDTransformer(max_features=20000, n_components=512)
print("Fitting TF-IDF + SVD transformer...")
X_train = transformer.fit_transform(train_texts)
X_val = transformer.transform(val_texts)

# Load MLP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_size=X_train.shape[1], output_size=len(LABEL_COLS), dropout=0.2)
model.load_state_dict(torch.load(MLP_PTH, map_location=device))
model.to(device)
model.eval()

# Predict
with torch.no_grad():
    X_val_t = torch.from_numpy(X_val).to(device)
    logits = model(X_val_t).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

# Per-class metrics
per_class = {}
for i, label in enumerate(LABEL_COLS):
    y_true = val_labels[:, i]
    y_pred = preds[:, i]
    per_class[label] = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred))
    }

# Binary metrics (any label)
binary_true = (val_labels.sum(axis=1) > 0).astype(int)
binary_pred = (preds.sum(axis=1) > 0).astype(int)
binary_probs = probs.max(axis=1)

acc = accuracy_score(binary_true, binary_pred)
prec = precision_score(binary_true, binary_pred, zero_division=0)
rec = recall_score(binary_true, binary_pred, zero_division=0)
f1 = f1_score(binary_true, binary_pred, zero_division=0)
try:
    auc = roc_auc_score(binary_true, binary_probs)
except Exception:
    auc = None

cm = confusion_matrix(binary_true, binary_pred)
tn, fp, fn, tp = cm.ravel()

metrics = {
    'model': MLP_PTH,
    'sample_fraction': SAMPLE_FRAC,
    'validation_samples': int(len(val_texts)),
    'binary': {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc) if auc is not None else None,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    },
    'per_class': per_class
}

out_path = 'mlp_train_sample_eval.json'
with open(out_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print(f"Results saved to {out_path}")
