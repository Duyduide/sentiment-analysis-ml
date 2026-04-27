# 🎯 Sentiment Analysis — Traditional ML vs. Modern NLP

A comparative study of **Traditional Machine Learning** (TF-IDF + Logistic Regression / Naive Bayes ) versus **Modern NLP** (DistilBERT embeddings + Neural Network) for sentiment classification.

> **Course:** Machine Learning CO3117 — HCMUT (Semester 252)

---

## 👥 Team Information

| Name | Student ID |
|------|-----------|
| Nguyễn Phương Duy | 2210526 | 
| Lê Khắc Dũng | 2210574 | 
| Nguyễn Hà Viết Thống | 2313339 | 
| Quách Trọng Kiên | 2211734 |

**Instructor:** *Ph.D Trương Vĩnh Lân*

---

## 📂 Project Structure

```
assignment/
├── modules/                  # Core Python modules
│   ├── __init__.py
│   ├── preprocessing.py      # Text cleaning & normalization
│   ├── features.py           # TF-IDF & Bag-of-word vs distilBERT feature extraction
│   ├── models.py             # Traditional ML + PyTorch MLP
│   └── utils.py              # Save/load features, plotting
├── notebooks/
│   └── main.ipynb            # Master Colab notebook (end-to-end pipeline)
├── reports/                  # Generated reports & figures
├── features/                 # Saved feature files (.npy)
└── requirements.txt &.ymal   # Python dependencies
```

---

## 🚀 Quick Start (Google Colab)

### 1. Clone & Install

```python
!git clone https://github.com/Duyduide/sentiment-analysis-ml.git
%cd sentiment-analysis-ml
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
- **Features:** TF-IDF + Bag-of-Words (BoW)
- **Models:** Logistic Regression, Naive Bayes

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
