# Natural Language Understanding

This repository contains implementations of core Natural Language Processing (NLP) techniques, focusing on **Word Embeddings** and **Sequence Modeling using RNNs**.

The projects are built from scratch as well as using optimized libraries to provide both **theoretical understanding** and **practical performance insights**.

---

## 📂 Projects Overview

### 🔹 1. Learning Word Embeddings from IIT Jodhpur Data

* Implementation of **Word2Vec models** on a custom IIT Jodhpur corpus
* Two approaches:

  * Gensim (optimized library-based)
  * From-scratch NumPy implementation
* Architectures:

  * CBOW (Continuous Bag of Words)
  * Skip-gram with Negative Sampling
* Includes:

  * Custom preprocessing pipeline
  * Hyperparameter tuning
  * Visualization (PCA, t-SNE)
  * Evaluation using cosine similarity

📁 Detailed implementation:
➡️ `LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA/README.md`

---

### 🔹 2. Character-Level Name Generation using RNN Variants

* Character-level language modeling for generating Indian full names

* Models implemented **from scratch in PyTorch**:

  * Vanilla RNN
  * Bidirectional LSTM (BLSTM)
  * Attention-based RNN (Bahdanau Attention)

* Includes:

  * Custom dataset pipeline
  * Training & generation scripts
  * Evaluation metrics:

    * Novelty Rate
    * Diversity
  * Model comparison & analysis

📁 Detailed implementation:
➡️ `CHARACTER-LEVEL NAME GENERATION USING RNN/README.md`

---

## 🚀 Setup & Installation

```bash
git clone https://github.com/Parth-Pitrubhakta/Natural-Language-Understanding.git
cd Natural-Language-Understanding
```

### Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Word Embeddings Project

```bash
cd "LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA"
```

Run Gensim pipeline:

```bash
python training/rebuild_and_answer.py
```

Run from-scratch implementation:

```bash
python scratch_w2v/word2vec_scratch.py
```

---

### 🔹 RNN Name Generation Project

```bash
cd "CHARACTER-LEVEL NAME GENERATION USING RNN"
```

Train models:

```bash
python train.py
```

Evaluate models:

```bash
python evaluate.py
```

---

## 📊 Key Concepts Covered

* Word Embeddings (CBOW, Skip-gram)
* Negative Sampling
* Cosine Similarity Evaluation
* Recurrent Neural Networks (RNN)
* LSTM & Bidirectional LSTM
* Attention Mechanisms (Bahdanau)
* Sequence Modeling
* Text Generation

---

## 📌 Repository Structure

```text
Natural-Language-Understanding/
│
├── LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA/
│   └── README.md
│
├── CHARACTER-LEVEL NAME GENERATION USING RNN/
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

## 🎯 Key Highlights

* ✔️ End-to-end NLP pipelines (preprocessing → training → evaluation)
* ✔️ From-scratch implementations for deep understanding
* ✔️ Comparison with optimized libraries (Gensim, PyTorch)
* ✔️ Custom dataset usage (IIT Jodhpur corpus & Indian names)
* ✔️ Quantitative + qualitative evaluation

---

## 🚀 Future Improvements

* Add pretrained embeddings (GloVe, FastText)
* Deploy models via web interface
* Expand datasets for better generalization
* Hyperparameter optimization automation

---

## 👨‍💻 Author

Parth Pitrubhakta
