import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random


class TfidfSVDTransformer:
    """Transforms text to dense features using TF-IDF then TruncatedSVD.

    This keeps memory usage reasonable for large vocabularies.
    """

    # CHANGE THESE VALUES TO TUNE FEATURE EXTRACTION
    def __init__(self, max_features=20000, n_components=512):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

    def fit_transform(self, texts):
        X = self.tfidf.fit_transform(texts)
        Xr = self.svd.fit_transform(X)
        return Xr.astype(np.float32)

    def transform(self, texts):
        X = self.tfidf.transform(texts)
        Xr = self.svd.transform(X)
        return Xr.astype(np.float32)


class ToxicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, output_size)
        )
        
    def forward(self, x):
        x = self.sequential(x)

        return x


def training(train_exs, dev_exs):
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get training data
    X_train, y_train = train_exs
    X_dev, y_dev = dev_exs

    # Hyperparametrs - PLAY AROUND WITH THESE VALUES
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    epochs = 5
    dropout = 0.2
    lr=1e-3

    # Initialize model
    model = MLP(input_size=input_size, output_size=output_size, dropout=dropout)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to tensors and DataLoaders
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_dev_t = torch.from_numpy(X_dev)
    y_dev_t = torch.from_numpy(y_dev)

    train_dataset = ToxicDataset(X_train_t, y_train_t)
    dev_dataset = ToxicDataset(X_dev_t, y_dev_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # Training Loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation Loop
        model.eval()
        dev_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                dev_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.cpu().numpy())

        dev_loss = dev_loss / len(dev_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Compute metrics
        f1 = f1_score(all_labels, all_preds, average='micro')
        dev_accuracy = (all_preds == all_labels).mean()
        exact_match = np.mean(np.all(all_preds == all_labels, axis=1))

        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={avg_loss:.4f} | " 
              f"dev_loss={dev_loss:.4f} | "
              f"dev_f1={f1} | "
              f"dev_acc={dev_accuracy} | "
              f"exact_match={exact_match}")