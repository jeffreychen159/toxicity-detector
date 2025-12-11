import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class ToxicTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def load_jigsaw(csv_path: Path, text_col: str, sample_frac: float = 1.0):
    df = pd.read_csv(csv_path)
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected label columns: {missing}")
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in CSV")

    # Apply sampling if requested
    if sample_frac < 1.0:
        # Stratified sampling to maintain class distribution
        has_toxic = df[LABEL_COLS].sum(axis=1) > 0
        df = df.groupby(has_toxic, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        ).reset_index(drop=True)
        print(f"Sampled {len(df)} examples ({sample_frac*100:.1f}% of original dataset)")

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[LABEL_COLS].astype(np.float32).values
    return texts, labels


def compute_metrics_builder(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.astype(np.float32)
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(np.int32)

        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
        macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
        exact_match = (preds == labels).all(axis=1).mean()
        try:
            roc_macro = roc_auc_score(labels, probs, average="macro")
        except ValueError:
            roc_macro = float("nan")

        metrics = {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "exact_match": exact_match,
            "roc_auc_macro": roc_macro,
        }
        for idx, label in enumerate(LABEL_COLS):
            metrics[f"f1_{label}"] = per_class_f1[idx]
        return metrics

    return compute_metrics


def per_label_confusion(labels: np.ndarray, preds: np.ndarray):
    output = {}
    for idx, label in enumerate(LABEL_COLS):
        cm = confusion_matrix(labels[:, idx], preds[:, idx], labels=[0, 1]).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
        else:
            # Handle degenerate cases where one class is missing
            tn = fp = fn = tp = 0
        output[label] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return output


def train_and_evaluate(args):
    texts, labels = load_jigsaw(Path(args.train_csv), args.text_col, args.sample_frac)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_size, random_state=42, stratify=labels.sum(axis=1) > 0
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

    # Use dynamic padding instead of pre-padding - much faster!
    train_encodings = tokenizer(train_texts, truncation=True, max_length=args.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=args.max_length)

    train_dataset = ToxicTextDataset(train_encodings, train_labels)
    val_dataset = ToxicTextDataset(val_encodings, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,  # Set to 0 for Colab
        save_total_limit=2,
        warmup_steps=args.warmup_steps,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(args.threshold),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # Detailed metrics and confusion matrices on the validation set
    preds_output = trainer.predict(val_dataset)
    logits = preds_output.predictions
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(np.int32)
    labels_np = preds_output.label_ids.astype(np.int32)

    confusions = per_label_confusion(labels_np, preds)
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0).tolist()
    try:
        per_class_roc = roc_auc_score(labels_np, probs, average=None).tolist()
    except ValueError:
        per_class_roc = [float("nan")] * len(LABEL_COLS)

    extras = {
        "per_class_f1": dict(zip(LABEL_COLS, per_class_f1)),
        "per_class_roc_auc": dict(zip(LABEL_COLS, per_class_roc)),
        "confusion_matrices": confusions,
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics | extras, f, indent=2)

    print("Validation metrics:")
    print(json.dumps(eval_metrics, indent=2))
    print("Per-class F1 / ROC-AUC and confusion matrices saved to eval_metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune HateBERT for Jigsaw multi-label toxicity.")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to Jigsaw train CSV.")
    parser.add_argument("--text_col", type=str, default="comment_text", help="Text column name.")
    parser.add_argument("--model_name", type=str, default="GroNLP/hateBERT", help="HF model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="outputs/hatebert", help="Where to save models/metrics.")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenizer.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Per-device train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Per-device eval batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold after sigmoid.")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of data to sample (0.0-1.0). Use 1.0 for full dataset.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16) training.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps.")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
