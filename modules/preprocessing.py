"""
preprocessing.py — Text Cleaning & Preprocessing Module
========================================================
Functions for text normalization, stopword removal, and tokenization
designed for sentiment analysis tasks.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK Resource Download (safe to call multiple times)
# ---------------------------------------------------------------------------

def download_nltk_resources():
    """Download required NLTK data packages."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Individual Cleaning Functions
# ---------------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<.*?>", "", text)


def remove_special_characters(text: str) -> str:
    """Remove special characters and digits, keeping only letters and spaces."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str, stop_words: set = None) -> str:
    """
    Remove English stopwords from text.

    Parameters
    ----------
    text : str
        Input text.
    stop_words : set, optional
        Custom stopword set. Defaults to NLTK English stopwords.

    Returns
    -------
    str
        Text with stopwords removed.
    """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize words in text using WordNet lemmatizer.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)


# ---------------------------------------------------------------------------
# Combined Pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(text: str, use_lemmatization: bool = True) -> str:
    """
    Apply the full text-cleaning pipeline.

    Steps: lowercase → remove URLs → remove HTML → remove special chars →
           remove extra whitespace → remove stopwords → (optional) lemmatize.

    Parameters
    ----------
    text : str
        Raw input text.
    use_lemmatization : bool
        Whether to apply lemmatization (default True).

    Returns
    -------
    str
        Cleaned text ready for feature extraction.
    """
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = remove_stopwords(text)

    if use_lemmatization:
        text = lemmatize_text(text)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "cleaned_text",
    use_lemmatization: bool = True,
) -> pd.DataFrame:
    """
    Apply the preprocessing pipeline to a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_col : str
        Name of the column containing raw text.
    output_col : str
        Name for the new cleaned-text column.
    use_lemmatization : bool
        Whether to apply lemmatization.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional cleaned-text column.
    """
    download_nltk_resources()
    df = df.copy()
    df[output_col] = df[text_col].apply(
        lambda x: preprocess_pipeline(x, use_lemmatization=use_lemmatization)
    )
    return df
