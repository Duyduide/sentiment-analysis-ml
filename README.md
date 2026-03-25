# 🎯 Sentiment Analysis — Traditional ML vs. Modern NLP

A comparative study of **Traditional Machine Learning** (TF-IDF + SVM / Naive Bayes / Logistic Regression) versus **Modern NLP** (DistilBERT embeddings + Neural Network) for sentiment classification.

> **Course:** Machine Learning — HCMUT (Semester 252)

---

## 👥 Team Information

| Name | Student ID | Role |
|------|-----------|------|
| Member 1 | XXXXXXX | Team Lead / ML Engineer |
| Member 2 | XXXXXXX | Data Engineer |
| Member 3 | XXXXXXX | NLP Engineer |

**Instructor:** *[Instructor Name]*

---

## 📂 Project Structure

```
assignment/
├── modules/                  # Core Python modules
│   ├── __init__.py
│   ├── preprocessing.py      # Text cleaning & normalization
│   ├── features.py           # TF-IDF & DistilBERT feature extraction
│   ├── models.py             # Traditional ML + PyTorch MLP
│   └── utils.py              # Save/load features, plotting
├── notebooks/
│   └── main.ipynb            # Master Colab notebook (end-to-end pipeline)
├── reports/                  # Generated reports & figures
├── features/                 # Saved feature files (.npy)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start (Google Colab)

### 1. Clone & Install

```python
!git clone https://github.com/<your-username>/<your-repo>.git /content/project
%cd /content/project
!pip install -r requirements.txt
```

### 2. Download Dataset via Kaggle API

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

!kaggle datasets download -d kazanova/sentiment140 -p /content/data --unzip
```

### 3. Run the Pipeline

Open `notebooks/main.ipynb` and execute the cells sequentially:

1. **Load & Explore** the dataset
2. **Preprocess** text (cleaning, stopword removal, lemmatization)
3. **Extract Features** (TF-IDF and DistilBERT embeddings)
4. **Train Models** (Traditional ML + Simple NN)
5. **Evaluate & Compare** results with confusion matrices and loss curves

---

## 🧪 Methods

### Traditional ML Pipeline
- **Features:** TF-IDF (unigram + bigram, max 5000 features)
- **Models:** Logistic Regression, Linear SVM, Multinomial Naive Bayes

### Modern NLP Pipeline
- **Features:** DistilBERT `[CLS]` token embeddings (768-dim)
- **Model:** 2-layer MLP (PyTorch) with dropout

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Traditional ML models & TF-IDF |
| `transformers` | DistilBERT tokenizer & model |
| `torch` | Neural network training |
| `nltk` | Text preprocessing |
| `matplotlib` / `seaborn` | Visualization |
| `kaggle` | Dataset download API |

See [`requirements.txt`](requirements.txt) for the full list.

---

## 📊 Expected Outputs

- Accuracy comparison table across all models
- Confusion matrices for each model
- Training / validation loss curves for the neural network
- Saved features in `features/` directory (`.npy` format)

---

## 📝 License

This project is for educational purposes as part of the HCMUT Machine Learning course.
