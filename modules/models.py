"""
models.py — Model Definitions & Training Module
================================================
Traditional ML models (Logistic Regression, SVM, Naive Bayes) and a simple
PyTorch MLP for BERT-embedding-based classification.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


# ======================================================================
# Traditional ML Models
# ======================================================================

class TraditionalModels:
    """Factory and convenience wrapper for scikit-learn classifiers."""

    @staticmethod
    def get_models() -> dict:
        """
        Return a dictionary of model name → estimator instance.

        Returns
        -------
        dict[str, estimator]
        """
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM (Linear)": SVC(kernel="linear", random_state=42),
            "Naive Bayes": MultinomialNB(),
        }

    @staticmethod
    def train_evaluate(model, X_train, X_test, y_train, y_test):
        """
        Fit a model and return evaluation metrics.

        Parameters
        ----------
        model : sklearn estimator
        X_train, X_test : array-like
        y_train, y_test : array-like

        Returns
        -------
        dict
            Keys: ``model``, ``accuracy``, ``report`` (classification report str).
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "y_pred": y_pred,
        }


# ======================================================================
# Simple Neural Network (PyTorch)
# ======================================================================

class SimpleNN(nn.Module):
    """
    A 2-layer MLP for classification on dense embeddings (e.g. BERT).

    Architecture
    ------------
    Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear (logits)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_nn(
    model: SimpleNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = None,
):
    """
    Train a SimpleNN model and return training history + predictions.

    Parameters
    ----------
    model : SimpleNN
    X_train, y_train : np.ndarray
    X_test, y_test : np.ndarray
    epochs : int
    batch_size : int
    learning_rate : float
    device : str or None
        ``"cuda"`` / ``"cpu"``; auto-detected if None.

    Returns
    -------
    dict
        Keys: ``model``, ``accuracy``, ``report``, ``y_pred``,
        ``train_losses``, ``val_losses``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)

    # --- DataLoaders -------------------------------------------------------
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Optimizer & Loss --------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    # --- Training Loop -----------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Validation ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"  Epoch {epoch:>3}/{epochs} — train_loss: {avg_train_loss:.4f}  val_loss: {avg_val_loss:.4f}")

    # --- Final Evaluation --------------------------------------------------
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "y_pred": y_pred,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
