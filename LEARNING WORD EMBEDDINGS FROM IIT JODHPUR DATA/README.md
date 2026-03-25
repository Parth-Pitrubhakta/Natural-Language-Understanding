# Word2Vec Implementation: Gensim vs. From-Scratch (IIT Jodhpur Corpus)

This repository contains the implementation of Word2Vec models trained on a custom, cleaned corpus of text collected from IIT Jodhpur sources (official website, academic regulations, syllabi, NIRF data). 

The project explores two distinct approaches:
1. **Gensim Library Execution:** Utilizing the highly optimized `gensim` library for high-speed training and evaluation.
2. **From-Scratch NumPy Execution:** A complete manual implementation of the Continuous Bag of Words (CBOW) and Skip-gram architectures, complete with Negative Sampling and custom gradient descent updates, using only raw NumPy (`scratch_w2v/word2vec_scratch.py`).

Both approaches are evaluated on domain-specific semantic analogies and word similarity metrics. 

---

## 🚀 Setup & Installation

**1. Clone the repository and navigate to the directory:**
```bash
git clone <your-repo-link>
cd iitj_word2vec
```

**2. Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On MacOS/Linux
# .\venv\Scripts\activate # On Windows
```

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```
*(Note: NLTK will automatically download the `punkt` and `stopwords` packages upon first execution).*

---

## 🏃‍♂️ How to Run the Code

### 1. The "From-Scratch" NumPy Pipeline
This script runs the custom NumPy implementation of Word2Vec. It processes the cleaned corpus, builds the vocabulary, initializes the input and output weight matrices, and trains the models using negative sampling and linear learning rate decay.

```bash
python scratch_w2v/word2vec_scratch.py
```

### 2. The Gensim (Library) Pipeline
This is the master script to execute the complete standard NLP pipeline using `gensim`. It runs a grid search over 24 different hyperparameter combinations.

```bash
python training/rebuild_and_answer.py
```

### 3. Run Preprocessing Only
```bash
python preprocessing/preprocess.py
```

---

## 📁 Repository Structure

```text
iitj_word2vec/
│
├── data/
│   ├── raw/                 # Raw HTML/Text collected from IITJ
│   ├── processed/           # 'corpus.txt' goes here
│   └── models/              # Saved Gensim Word2Vec models
│
├── preprocessing/
│   └── preprocess.py        # Text cleaning, NLTK tokenization, RegExp 
│
├── training/
│   ├── train_word2vec.py      # Core Gensim model training algorithms
│   └── rebuild_and_answer.py  # Master script for the library pipeline
│
├── scratch_w2v/
│   ├── word2vec_scratch.py  # The pure NumPy Word2Vec implementation
│   └── models/              # Saved numpy coordinate arrays (.npz)
│
├── outputs/                 
│   ├── plots/               # Visualizations (Loss curves, PCA, t-SNE)
│   └── results/             # JSON metrics showing model performance
```

---

## 📊 Comprehensive Output Report

### Q1: Preprocessing Steps
1. Collected textual data from IIT Jodhpur sources.
2. Removed boilerplate text (recurring footers/headers, navigation artifacts).
3. Removed URLs, email addresses, and phone numbers using regex.
4. Lowercased all text for uniform representation.
5. Removed non-ASCII characters (filters Hindi/regional text).
6. Removed excessive punctuation; kept only alphabetic tokens of length > 1.
7. Tokenized text into sentences (`sent_tokenize`), then words (`word_tokenize`).
8. Removed English stop words (the, of, and, etc.) using NLTK's list.
9. Filtered out very short sentences (< 3 tokens).
10. Saved cleaned corpus as `corpus.txt`.

### Q2: Example Embedding Vector ("research")
Word: **research** (100-dimensional from our best CBOW NumPy implementation)
```
-0.4677, -1.1765, -0.3751, 0.5148, 0.1460, 1.9333, 1.2102, -1.8994, 0.5372, 0.8875, 0.2795, -0.5373, 1.1289, -0.0472, -1.1088, 1.4598, -0.1094, -0.8982, -0.7603, 0.0588, 1.4337, 0.8710, 0.1872, 1.2729, -1.4659, 0.3241, -0.7062, 2.2498, -0.2489, -1.6895, 0.8866, 0.5213, 0.5265, -0.0903, 0.5990, 0.2779, -1.1381, 1.3741, 0.6142, -0.5515, -0.3450, -1.2204, 0.6251, 0.7653, -0.9227, -0.3341, -1.4072, -0.5218, 0.4388, 0.8945, -0.7525, 0.0417, 0.8766, -0.8397, 0.3306, 0.3259, -0.9294, -1.3566, -1.6705, -0.2013, 0.1157, 1.6415, -1.0336, 0.5469, 1.8630, 0.1111, 0.1957, -0.4000, 1.6346, -1.6778, -0.1974, -0.5378, -0.2767, 0.5423, -0.8614, 0.4249, 1.8570, 0.5747, -1.5963, -1.4708, -0.1551, 0.3848, -1.2825, 0.2341, 0.5457, -0.7077, -0.8720, -0.3754, -0.7886, 0.2038, -0.4861, -0.4243, 0.3161, -1.4693, 0.6961, -1.9869, 0.2635, 0.7843, -1.0112, 1.1116
```

### Q3: Top-10 Words (frequency-wise, stop words removed)
`jodhpur: 2987`, `professor: 2794`, `yes: 2572`, `regular: 2281`, `iit: 2270`, `research: 2034`, `male: 1946`, `assistant: 1864`, `india: 1287`, `engineering: 1282`

### Q4: Interesting Analogies Learned
- **UG : BTech :: PG : ?** $\rightarrow$ `last (0.9206)`, `syllabus (0.8927)`
- **student : exam :: professor : ?** $\rightarrow$ `dean (0.7920)`
- **research : PhD :: teaching : ?** $\rightarrow$ `mba (0.3771)`

### Nearest Neighbors Examples
- **student**: `pursuing (0.9382)`, `strength (0.9256)`, `ug (0.9198)`, `admitted (0.9036)`, `pg (0.9025)`
- **exam**: `round (0.9575)`, `test (0.9566)`, `ii (0.9540)`, `written (0.9518)`, `provisionally (0.9496)`

---

## 📈 Evaluation Metric (`Eval Score`)

Because Word2Vec is an unsupervised algorithm without standard accuracy metrics, we evaluate models based on how well they capture domain-specific semantic relationships. The `Eval Score` is calculated as the **average cosine similarity** across 8 academic word pairs tightly relevant to the IIT Jodhpur corpus (e.g., `research/laboratory`, `student/faculty`).

### Comparison Table

| Model Environment | Architecture | Best Config | Eval Score | Training Time |
|-------------------|--------------|-------------|------------|---------------|
| **Gensim (Library)** | CBOW | dim=200, win=5, neg=5 | 0.6231 | ~1.17s |
| **Gensim (Library)** | Skip-gram | dim=100, win=3, neg=5 | 0.5027 | ~1.81s |
| **From-Scratch (NumPy)** | CBOW | dim=100, win=5, neg=5 | **0.6403** | ~143.39s |
| **From-Scratch (NumPy)** | Skip-gram | dim=100, win=3, neg=5 | 0.4681 | ~220.07s |

![Model Comparison](outputs/plots/scratch_vs_gensim_comparison.png)

### Key Observations
1. **Gensim is significantly faster** — uses Cython-optimized C extensions & multi-threading, whereas our scratch implementation uses pure Python loops + NumPy.
2. **From-Scratch model captures nuances effectively** — our raw NumPy CBOW model actually hit a slightly higher Eval Score on our specialized domain pairs (0.6403) than the equivalent Gensim model, showing that the negative-sampling loss equations and gradient updates were implemented robustly.
3. **Architecture trends are consistent** — CBOW consistently outperforms Skip-gram on this small-scale corpus in both the Gensim and Manual implementations.

---

### Implementation Details: NumPy Word2Vec
Our manual implementation (`scratch_w2v/word2vec_scratch.py`) handles:
* Vocabulary building with dynamic frequency filtering (`min_count=3`).
* **Negative sampling** utilizing the standard unigram distribution raised to the $3/4$ power: $P(w)^{0.75}$.
* Vectorized forward and backward passes calculating custom analytical gradients ($\frac{\partial L}{\partial W_{in}}$ and $\frac{\partial L}{\partial W_{out}}$).
* Stochastic Gradient Descent (SGD) with linear learning rate decay.
