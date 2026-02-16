# ml/lstm_model.py  (BINARY IMBALANCE PREDICTION)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------- MODEL ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)  # binary output

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # last timestep
        out = self.fc(out)
        return out.squeeze(-1)  # (batch,)


def run_lstm(
    data_path="ml/lob_sequences.npz",
    epochs=12,
    batch_size=32,
    lr=1e-3
):
    # ---------------- LOAD DATA ----------------
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]  # already 0/1

    # ---------------- TRAIN / TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- TORCH DATA ----------------
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    # ---------------- MODEL ----------------
    input_dim = X.shape[2]
    model = LSTMClassifier(input_dim=input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------- TRAIN ----------------
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")

    # ---------------- EVALUATE ----------------
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().numpy()

    acc = accuracy_score(y_test.numpy(), preds)

    print("\n==== LSTM (IMBALANCE) RESULTS ====")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test.numpy(), preds, digits=4))
    print("=================================")


if __name__ == "__main__":
    run_lstm()
