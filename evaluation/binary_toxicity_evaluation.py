# Disable wandb
import os
os.environ['WANDB_DISABLED'] = 'true'
 
# Subtask A: Binary Toxicity Classification Evaluation
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
import json
 
# Evaluate BERTweet model on 10% sample of train.csv
bertweet_pth = 'bertweet_model.pth'
try:
    from bertweet import BertTweetClassifier, TOKENIZER_NAME
except Exception:
    BertTweetClassifier = None
    TOKENIZER_NAME = None

if os.path.exists(bertweet_pth) and BertTweetClassifier is not None and TOKENIZER_NAME is not None:
    print(f"Found BERTweet model state: {bertweet_pth}. Running BERTweet evaluation...")

    df = pd.read_csv('train.csv')
    LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Sample 10% for evaluation
    sample_frac = 1
    df_sample = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    texts = df_sample['comment_text'].fillna('').astype(str).tolist()
    labels = df_sample[LABEL_COLS].values

    # Split out a small validation set (10% of the sampled data)
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=(labels.sum(axis=1) > 0)
    )

    print(f"Sampled {len(df_sample):,} rows (frac={sample_frac}); validation size: {len(val_texts):,}")

    # Load BERTweet model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BertTweetClassifier(output_size=6, dropout=0.2)
    model.load_state_dict(torch.load(bertweet_pth, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)
    print("Tokenizing validation texts...")
    
    val_encodings = tokenizer(val_texts, truncation=True, max_length=128, padding=True, return_tensors='pt')

    # Batch prediction
    print("Running predictions...")
    all_logits = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_input_ids = val_encodings['input_ids'][i:i+batch_size].to(device)
            batch_attention_mask = val_encodings['attention_mask'][i:i+batch_size].to(device)
            logits = model(batch_input_ids, batch_attention_mask)
            all_logits.append(logits.cpu().numpy())

    logits = np.vstack(all_logits)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    # Convert to binary (any label positive)
    binary_true = (val_labels.sum(axis=1) > 0).astype(int)
    binary_pred = (preds.sum(axis=1) > 0).astype(int)
    binary_probs = probs.max(axis=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    acc = accuracy_score(binary_true, binary_pred)
    prec = precision_score(binary_true, binary_pred, zero_division=0)
    rec = recall_score(binary_true, binary_pred, zero_division=0)
    f1 = f1_score(binary_true, binary_pred, zero_division=0)
    try:
        auc = roc_auc_score(binary_true, binary_probs)
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(binary_true, binary_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'model': bertweet_pth,
        'sample_fraction': sample_frac,
        'validation_samples': int(len(val_texts)),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc) if not np.isnan(auc) else None,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'support': {'non_toxic': int((binary_true == 0).sum()), 'toxic': int((binary_true == 1).sum())}
    }

    print("\n" + "="*60)
    print("SUBTASK A: BINARY TOXICITY CLASSIFICATION (BERTweet)")
    print("="*60)
    print(json.dumps(metrics, indent=2))

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")

    save_path = 'mlp_binary_classification_results.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    # Exit after MLP evaluation
    raise SystemExit(0)
else:
    print("No BERTweet model found. Please ensure bertweet_model.pth exists.")