"""
utils.py — Utility Functions
=============================
Helpers for saving/loading features and plotting evaluation results.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ======================================================================
# Feature Persistence (NumPy .npy)
# ======================================================================

def save_features(array: np.ndarray, path: str) -> None:
    """
    Save a numpy array to disk in ``.npy`` format.

    Parameters
    ----------
    array : np.ndarray
        Feature array to save.
    path : str
        Destination file path (should end with ``.npy``).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
    print(f"[utils] Features saved → {path}  (shape: {array.shape})")


def load_features(path: str) -> np.ndarray:
    """
    Load a numpy array from a ``.npy`` file.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file.

    Returns
    -------
    np.ndarray
    """
    array = np.load(path, allow_pickle=True)
    print(f"[utils] Features loaded ← {path}  (shape: {array.shape})")
    return array


# ======================================================================
# Visualization
# ======================================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list[str], optional
        Display labels for the axes.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Figure saved → {save_path}")

    plt.show()


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] = None,
    title: str = "Training Loss Curve",
    save_path: str = None,
) -> None:
    """
    Plot training (and optionally validation) loss over epochs.

    Parameters
    ----------
    train_losses : list[float]
        Per-epoch training loss values.
    val_losses : list[float], optional
        Per-epoch validation loss values.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2)
    if val_losses is not None:
        plt.plot(epochs, val_losses, "s--", label="Val Loss", linewidth=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Figure saved → {save_path}")

    plt.show()
