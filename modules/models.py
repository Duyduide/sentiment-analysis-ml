"""
models.py — Model Training & Evaluation Module
===============================================
Wrappers for training scikit-learn classifiers and evaluating sentiment
analysis models (TF-IDF/BoW-based and BERT-embedding-based).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
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
    y_proba=None,
) -> dict:
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
    y_proba : array-like, optional
        Predicted class probabilities (n_samples, n_classes). When provided,
        macro-averaged ROC-AUC (One-vs-Rest) is computed and included.

    Returns
    -------
    dict
        {'Accuracy': float, 'Macro F1': float, 'Weighted F1': float}
        plus 'ROC-AUC' when y_proba is supplied.
    """
    print(f"===== {model_name} =====")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy:        {acc:.4f}")

    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    metrics = {"Accuracy": acc, "Macro F1": macro_f1, "Weighted F1": weighted_f1}

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            print(f"ROC-AUC (macro): {auc:.4f}")
            metrics["ROC-AUC"] = auc
        except Exception as exc:
            print(f"ROC-AUC skipped: {exc}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        title=f"{model_name} — Confusion Matrix",
    )

    return metrics


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict, title: str = "Model Comparison") -> pd.DataFrame:
    """
    Build a multi-metric comparison table and grouped bar-chart.

    Parameters
    ----------
    results : dict
        Mapping model_name → dict of metrics (from evaluate_model) or a plain
        float (legacy, treated as Accuracy).
    title : str
        Chart title.

    Returns
    -------
    pd.DataFrame
        Sorted comparison table with columns ['Model', metric, ...].
    """
    rows = []
    for name, val in results.items():
        if isinstance(val, dict):
            rows.append({"Model": name, **val})
        else:
            rows.append({"Model": name, "Accuracy": float(val)})

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != "Model"]
    df = df.sort_values(metric_cols[0], ascending=False).reset_index(drop=True)

    df_melt = df.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(max(8, 1.2 * len(df)), 5))
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melt)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    plt.show()

    return df
