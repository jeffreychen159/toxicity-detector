# Cross-Domain Evaluation: Real Toxicity Prompts (Reddit)
import os
os.environ['WANDB_DISABLED'] = 'true'

import numpy as np
import pandas as pd
import torch
import json
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Import BertTweetClassifier for loading custom model
try:
    from bert import BertTweetClassifier, TOKENIZER_NAME
except ImportError:
    print("Warning: Could not import BertTweetClassifier from bert.py")

print("="*60)
print("CROSS-DOMAIN EVALUATION: REDDIT DATA")
print("="*60)

# Load Real Toxicity Prompts dataset
print("\n📥 Loading Real Toxicity Prompts dataset from HuggingFace...")
dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
print(f"✅ Loaded {len(dataset):,} samples")

# Convert to pandas for easier manipulation
df = pd.DataFrame(dataset)
 
# Extract prompts and toxicity scores
# The dataset has 'prompt' and 'continuation' fields with nested toxicity scores
print("\n🔍 Processing toxicity scores...")
 
def extract_toxicity(example):
    """Extract toxicity score from prompt or continuation"""
    # Try to get toxicity from prompt first
    if example.get('prompt') and isinstance(example['prompt'], dict):
        toxicity = example['prompt'].get('toxicity')
        text = example['prompt'].get('text', '')
    # Otherwise from continuation
    elif example.get('continuation') and isinstance(example['continuation'], dict):
        toxicity = example['continuation'].get('toxicity')
        text = example['continuation'].get('text', '')
    else:
        toxicity = None
        text = ''
 
    return {'text': text, 'toxicity_score': toxicity}
 
# Process all examples
processed = [extract_toxicity(ex) for ex in dataset]
reddit_df = pd.DataFrame(processed)
 
# Filter out examples without toxicity scores or empty text
reddit_df = reddit_df[reddit_df['toxicity_score'].notna()]
reddit_df = reddit_df[reddit_df['text'].str.len() > 0]
 
print(f"✅ Processed {len(reddit_df):,} samples with toxicity scores")
 
# Convert toxicity scores to binary labels (threshold at 0.5)
reddit_df['binary_label'] = (reddit_df['toxicity_score'] >= 0.5).astype(int)
 
print(f"\n📊 Dataset Statistics:")
print(f"  Total samples: {len(reddit_df):,}")
print(f"  Toxic samples: {reddit_df['binary_label'].sum():,} ({reddit_df['binary_label'].mean()*100:.1f}%)")
print(f"  Non-toxic samples: {(reddit_df['binary_label'] == 0).sum():,} ({(reddit_df['binary_label'] == 0).mean()*100:.1f}%)")
print(f"  Mean toxicity score: {reddit_df['toxicity_score'].mean():.3f}")
 
# Sample for faster evaluation (use 10,000 samples)
EVAL_SAMPLES = min(10000, len(reddit_df))
reddit_sample = reddit_df.sample(n=EVAL_SAMPLES, random_state=42).reset_index(drop=True)
print(f"\n✅ Sampled {EVAL_SAMPLES:,} examples for evaluation")

# Load the trained BERTweet model
print("\n📦 Loading trained BERTweet model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_loaded = False

# Try to load bertweet_model.pth
pth_path = 'mlp_model.pth'
if os.path.exists(pth_path):
    try:
        print(f"Loading {pth_path}...")
        model = BertTweetClassifier(output_size=6, dropout=0.2)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)
        model_loaded = True
        print(f"✅ Successfully loaded {pth_path}")
    except Exception as e:
        print(f"❌ Failed to load {pth_path}: {e}")

# If .pth model didn't load, try checkpoint directories
if not model_loaded:
    model_path = None
    for base_path in ['outputs/hatebert_full', 'outputs/hatebert_test']:
        if os.path.exists(base_path):
            checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
            if checkpoints:
                checkpoint_nums = [int(cp.split('-')[1]) for cp in checkpoints]
                best_checkpoint = f'checkpoint-{max(checkpoint_nums)}'
                model_path = f'{base_path}/{best_checkpoint}'
                print(f"Found checkpoint: {model_path}")
                break
            if os.path.exists(f'{base_path}/config.json'):
                model_path = base_path
                print(f"Found model config: {model_path}")
                break
    
    if model_path is None:
        print("❌ No trained model found (.pth or checkpoint directories). Please train BERTweet first.")
    else:
        print(f"✅ Loading from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        model.eval()
        model_loaded = True

if model_loaded:
    # Tokenize Reddit data
    print("\n🔤 Tokenizing Reddit text...")
    texts = reddit_sample['text'].tolist()
    labels = reddit_sample['binary_label'].values

    encodings = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors='pt')

    # Create simple batch processing
    print("\n🚀 Running predictions on Reddit data...")
    all_probs = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_input_ids = encodings['input_ids'][i:i+batch_size].to(device)
            batch_attention_mask = encodings['attention_mask'][i:i+batch_size].to(device)
            
            outputs = model(batch_input_ids, batch_attention_mask)
            logits = outputs  # Shape: (batch_size, 6)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    
    # For toxic detection, use the first label (toxic)
    toxic_probs = all_probs[:, 0]  # Shape: (num_samples,)
    preds = (toxic_probs >= 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, toxic_probs)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # Determine model path for reporting
    model_description = pth_path if os.path.exists(pth_path) else (model_path if 'model_path' in locals() else 'unknown')

    # Results
    reddit_metrics = {
        'dataset': 'Real Toxicity Prompts (Reddit)',
        'model': model_description,
        'samples_evaluated': EVAL_SAMPLES,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'support': {'non_toxic': int((labels == 0).sum()), 'toxic': int((labels == 1).sum())}
    }

    print("\n" + "="*60)
    print("CROSS-DOMAIN RESULTS: REDDIT TOXICITY DETECTION")
    print("="*60)
    print(json.dumps(reddit_metrics, indent=2))
 
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives:  {tp:,}")

    # Load Jigsaw results for comparison
    jigsaw_results_path = f'binary_classification_results.json'
    if os.path.exists(jigsaw_results_path):
        with open(jigsaw_results_path, 'r') as f:
            jigsaw_metrics = json.load(f)

        print("\n" + "="*60)
        print("DOMAIN GAP ANALYSIS: JIGSAW (Wikipedia) vs REDDIT")
        print("="*60)
        print(f"{'Metric':<20} {'Jigsaw (Wikipedia)':<20} {'Reddit':<20} {'Gap':<15}")
        print("-"*75)
        print(f"{'Accuracy':<20} {jigsaw_metrics['accuracy']:<20.4f} {acc:<20.4f} {jigsaw_metrics['accuracy']-acc:+.4f}")
        print(f"{'Precision':<20} {jigsaw_metrics['precision']:<20.4f} {prec:<20.4f} {jigsaw_metrics['precision']-prec:+.4f}")
        print(f"{'Recall':<20} {jigsaw_metrics['recall']:<20.4f} {rec:<20.4f} {jigsaw_metrics['recall']-rec:+.4f}")
        print(f"{'F1 Score':<20} {jigsaw_metrics['f1']:<20.4f} {f1:<20.4f} {jigsaw_metrics['f1']-f1:+.4f}")
        print(f"{'ROC-AUC':<20} {jigsaw_metrics['roc_auc']:<20.4f} {auc:<20.4f} {jigsaw_metrics['roc_auc']-auc:+.4f}")

        # Performance drop analysis
        f1_drop = jigsaw_metrics['f1'] - f1
        drop_pct = (f1_drop / jigsaw_metrics['f1']) * 100

        print(f"\n📉 Performance Drop:")
        print(f"  F1 Score decreased by {f1_drop:.4f} ({drop_pct:.1f}%)")

        if drop_pct < 5:
            print(f"  ✅ Excellent generalization! Model performs similarly on Reddit data.")
        elif drop_pct < 15:
            print(f"  ✓ Good generalization with minor domain gap.")
        else:
            print(f"  ⚠️ Significant domain gap detected. Model struggles with Reddit-style language.")

    # Save results
    save_path = f'reddit_cross_domain_results.json'
    with open(save_path, 'w') as f:
        json.dump(reddit_metrics, f, indent=2)
    print(f"\n✅ Results saved to: {save_path}")
