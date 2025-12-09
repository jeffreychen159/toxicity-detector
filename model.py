import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import time

class TfidfSVDTransformer:
    """Transforms text to dense features using TF-IDF then TruncatedSVD.

    This keeps memory usage reasonable for large vocabularies.
    """

    # CHANGE THESE VALUES TO TUNE FEATURE EXTRACTION
    def __init__(self, max_features=20000, n_components=512):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), stop_words='english')
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
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(512, output_size)
        
    def forward(self, x):
        x = self.sequential(x)
        x = self.output_layer(x)

        return x


def train_MLP(train_exs, dev_exs):
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get training data
    X_train, y_train = train_exs
    X_dev, y_dev = dev_exs

    # Hyperparametrs - PLAY AROUND WITH THESE VALUES
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print("Input Size: ", input_size)
    print("Output Size: ", output_size)

    epochs = 4
    dropout = 0.2
    lr=1e-3
    # Class labels for per-class metrics (Jigsaw categories)
    class_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

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
    
    epoch_list = []
    train_loss_list = []
    dev_loss_list = []

    dev_accuracy_list = []
    dev_f1_macro_list = []
    exact_match_list = []
    
    # Training Loop
    for epoch in range(1, epochs+1):
        start_time = time.time()
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

        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_dev_loss = dev_loss / len(dev_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Compute metrics
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        try:
            f1_macro = f1_score(all_labels, all_preds, average='macro')
        except Exception:
            f1_macro = None
        dev_accuracy = (all_preds == all_labels).mean()
        exact_match = np.mean(np.all(all_preds == all_labels, axis=1))

        # Per-class metrics
        try:
            f1_per_class = f1_score(all_labels, all_preds, average=None)
        except Exception:
            f1_per_class = [None] * all_labels.shape[1]
        accuracy_per_class = []
        for i in range(all_labels.shape[1]):
            try:
                acc = accuracy_score(all_labels[:, i], all_preds[:, i])
            except Exception:
                acc = None
            accuracy_per_class.append(acc)
        accuracy_per_class = np.array(accuracy_per_class, dtype=object)

        # Prepare formatted macro f1 string
        if f1_macro is None:
            f1_macro_str = "N/A"
        else:
            f1_macro_str = f"{f1_macro:.4f}"

        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={avg_loss:.4f} | " 
              f"dev_loss={avg_dev_loss:.4f} | "
              f"dev_f1_micro={f1_micro:.4f} | "
              f"dev_f1_macro={f1_macro_str} | "
              f"dev_acc={dev_accuracy:.4f} | "
              f"exact_match={exact_match:.4f} | "
              f"time={elapsed_time:.2f} seconds")
        # Print per-class metrics with labels (if available)
        for i in range(all_labels.shape[1]):
            label_name = class_labels[i] if i < len(class_labels) else f"class_{i}"
            f1_val = f1_per_class[i] if f1_per_class is not None else None
            acc_val = accuracy_per_class[i]
            f1_str = f"{f1_val:.4f}" if f1_val is not None else "N/A"
            acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
            print(f"  {label_name:15s} | F1: {f1_str} | Acc: {acc_str}")
        
        epoch_list.append(epoch)
        train_loss_list.append(avg_loss)
        dev_loss_list.append(avg_dev_loss)
        dev_accuracy_list.append(dev_accuracy)
        dev_f1_macro_list.append(f1_macro if f1_macro is not None else float('nan'))
        exact_match_list.append(exact_match)
    
    # Plotting results
    # plotting(epoch_list, train_loss_list, dev_loss_list, dev_accuracy_list, exact_match_list)

    # Save trained MLP model
    model_save_path = "mlp_model.pth"
    try:
        print(f"Saving MLP model to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved successfully to {model_save_path}")
    except Exception as e:
        print(f"Failed to save MLP model: {e}")

    return model

def plotting(epoch_list, train_loss_list, dev_loss_list, dev_accuracy_list, exact_match_list):
    plt.figure()

    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, dev_loss_list, label='Dev Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Dev Loss over Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epoch_list, dev_accuracy_list, label='Dev Accuracy')
    plt.plot(epoch_list, exact_match_list, label='Exact Match')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Dev Accuracy and Exact Match over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    