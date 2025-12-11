from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import time
import pandas as pd

TOKENIZER_NAME = "vinai/bertweet-base"
MODEL_NAME = "vinai/bertweet-base"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)
print("Tokenizer and model loaded.")

def encode_texts(texts, max_len=128):
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

class BertTweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }
    
class BertTweetClassifier(nn.Module):
    def __init__(self, output_size, dropout=0.2):
        super().__init__()
        self.bertweet = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.fc(x)
    
def train_bertweet(train_texts, train_labels, dev_texts, dev_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    epochs = 3
    lr = 1e-5
    
    # Define class labels (Jigsaw toxicity categories)
    class_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    print("Pre-tokenizing...")
    train_enc = encode_texts(train_texts)
    dev_enc = encode_texts(dev_texts)

    print("Preparing datasets and dataloaders...")
    train_dataset = BertTweetDataset(train_enc, train_labels)
    dev_dataset = BertTweetDataset(dev_enc, dev_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    print("Datasets and dataloaders prepared.")

    print("Initializing model...")
    model = BertTweetClassifier(output_size=train_labels.shape[1])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    print("Model initialized.")

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        total_train_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            total_train_samples += labels.size(0)

        avg_train_loss = total_train_loss / total_train_samples

        # ----- Validation -----
        model.eval()
        total_dev_loss = 0
        total_dev_samples = 0
        preds_list = []
        labels_list = []
        probs_list = []


        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_dev_loss += loss.item() * labels.size(0)
                total_dev_samples += labels.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.50).astype(int)

                preds_list.append(preds)
                probs_list.append(probs)
                labels_list.append(labels.cpu().numpy())

        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_dev_loss = total_dev_loss / total_dev_samples
        
        probs = np.vstack(probs_list)
        preds = np.vstack(preds_list)
        labels = np.vstack(labels_list)

        f1 = f1_score(labels, preds, average='micro')
        dev_accuracy = (preds == labels).mean()
        exact_match = np.mean(np.all(preds == labels, axis=1))
        
        # Per-class metrics
        f1_per_class = f1_score(labels, preds, average=None)
        accuracy_per_class = []
        for i in range(labels.shape[1]):
            acc = accuracy_score(labels[:, i], preds[:, i])
            accuracy_per_class.append(acc)
        accuracy_per_class = np.array(accuracy_per_class)
        
        # Print per-class F1 and accuracy
        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={avg_train_loss:.4f} | " 
              f"dev_loss={avg_dev_loss:.4f} | "
              f"dev_f1={f1:.4f} | "
              f"dev_acc={dev_accuracy:.4f} | "
              f"exact_match={exact_match:.4f} | "
              f"time={elapsed_time:.2f}s")
        # Print per-class metrics with labels
        for i, label in enumerate(class_labels):
            print(f"  {label:15s} | F1: {f1_per_class[i]:.4f} | Acc: {accuracy_per_class[i]:.4f}")
    
    # Save model at the end of training
    model_save_path = "bertweet_model.pth"
    print(f"Saving model to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    return model