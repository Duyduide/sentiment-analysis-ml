"""
models.py — Model Training & Evaluation Module
===============================================
Wrappers for training scikit-learn classifiers and evaluating sentiment
analysis models (both TF-IDF-based and BERT-embedding-based).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from modules.utils import plot_confusion_matrix


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def get_classifiers() -> dict:
    """
    Return a dict of default classifier instances ready for training.

    Returns
    -------
    dict
        Mapping model_name → unfitted sklearn estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced"),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, sample_weight=None):
    """
    Fit a sklearn estimator and return it.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model instance.
    X_train : array-like or sparse matrix
        Training features.
    y_train : array-like
        Training labels.
    sample_weight : array-like, optional
        Per-sample weights passed to ``model.fit``. Useful for models
        that do not support ``class_weight`` (e.g. MultinomialNB).

    Returns
    -------
    Fitted estimator.
    """
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_test,
    y_pred,
    model_name: str,
    labels: list = None,
) -> float:
    """
    Print accuracy + classification report and plot a confusion matrix.

    Parameters
    ----------
    y_test : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Display name shown in output and plot title.
    labels : list[str], optional
        Human-readable class names for the confusion matrix axes.

    Returns
    -------
    float
        Accuracy score.
    """
    print(f"===== {model_name} =====")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy:        {acc:.4f}")

    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f"{model_name} — Confusion Matrix",
    )

    return acc


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict, title: str = "Model Comparison") -> pd.DataFrame:
    """
    Build an accuracy comparison table and bar-chart.

    Parameters
    ----------
    results : dict
        Mapping model_name → accuracy (float).
    title : str
        Chart title.

    Returns
    -------
    pd.DataFrame
        Sorted comparison table with columns ['Model', 'Accuracy'].
    """
    df = pd.DataFrame(
        {"Model": list(results.keys()), "Accuracy": list(results.values())}
    )
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="Model", y="Accuracy", data=df)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return df
