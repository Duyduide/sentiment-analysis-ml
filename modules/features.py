"""
features.py — Feature Extraction Module
========================================
Provides TF-IDF, Bag-of-Words (BoW), and DistilBERT embedding extraction
for sentiment analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm


class FeatureExtractor:
    """
    Unified feature extractor supporting TF-IDF, BoW, and DistilBERT embeddings.

    Attributes
    ----------
    tfidf_vectorizer : TfidfVectorizer
        Scikit-learn TF-IDF vectorizer (fitted after calling ``extract_tfidf``).
    bow_vectorizer : CountVectorizer
        Scikit-learn BoW count vectorizer (fitted after calling ``extract_bow``).
    bert_tokenizer : DistilBertTokenizer or None
        HuggingFace tokenizer (loaded lazily).
    bert_model : DistilBertModel or None
        HuggingFace DistilBERT model (loaded lazily).
    device : torch.device
        CUDA if available, otherwise CPU.
    """

    def __init__(
        self,
        max_tfidf_features: int = 5000,
        max_bow_features: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        max_tfidf_features : int
            Maximum number of features for the TF-IDF vectorizer.
        max_bow_features : int or None
            Maximum number of features for the BoW vectorizer. If None,
            ``max_tfidf_features`` is reused so TF-IDF and BoW are comparable.
        """
        self.max_tfidf_features = max_tfidf_features
        self.max_bow_features = max_bow_features or max_tfidf_features

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.bow_vectorizer = CountVectorizer(
            max_features=self.max_bow_features,
            ngram_range=(1, 2),
        )
        self.bert_tokenizer = None
        self.bert_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------

    def extract_tfidf(self, corpus: list[str], fit: bool = True):
        """
        Extract TF-IDF features from a text corpus.

        Parameters
        ----------
        corpus : list[str]
            List of cleaned text documents.
        fit : bool
            If True, fit the vectorizer on the corpus (use for training).
            If False, only transform (use for test data).

        Returns
        -------
        scipy.sparse matrix
            TF-IDF feature matrix of shape (n_samples, max_tfidf_features).
        """
        if fit:
            return self.tfidf_vectorizer.fit_transform(corpus)
        else:
            return self.tfidf_vectorizer.transform(corpus)
        
    # ------------------------------------------------------------------
    # Bag-of-Words (BoW)
    # ------------------------------------------------------------------

    def extract_bow(self, corpus: list[str], fit: bool = True):
        """
        Extract Bag-of-Words count features from a text corpus.

        Parameters
        ----------
        corpus : list[str]
            List of cleaned text documents.
        fit : bool
            If True, fit the vectorizer on the corpus (use for training).
            If False, only transform (use for test data).

        Returns
        -------
        scipy.sparse matrix
            BoW count feature matrix of shape (n_samples, max_bow_features).
        """
        if fit:
            return self.bow_vectorizer.fit_transform(corpus)
        else:
            return self.bow_vectorizer.transform(corpus)

    # ------------------------------------------------------------------
    # DistilBERT Embeddings
    # ------------------------------------------------------------------

    def _load_bert(self):
        """Lazily load DistilBERT tokenizer and model."""
        if self.bert_tokenizer is None:
            print("[FeatureExtractor] Loading DistilBERT model...")
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert_model = DistilBertModel.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print(f"[FeatureExtractor] Model loaded on {self.device}")

    def extract_bert(
        self,
        corpus: list[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> np.ndarray:
        """
        Extract DistilBERT [CLS] token embeddings for a list of texts.

        Parameters
        ----------
        corpus : list[str]
            List of text documents (raw or cleaned).
        batch_size : int
            Number of texts per forward pass.
        max_length : int
            Maximum token length for the tokenizer.

        Returns
        -------
        np.ndarray
            Embeddings array of shape (n_samples, 768).
        """
        self._load_bert()

        all_embeddings = []

        for i in tqdm(range(0, len(corpus), batch_size), desc="Extracting BERT embeddings"):
            batch_texts = corpus[i : i + batch_size]
            encoded = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.bert_model(**encoded)

            # Use the [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.concatenate(all_embeddings, axis=0)
