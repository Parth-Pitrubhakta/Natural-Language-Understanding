# Character-Level Name Generation Using RNN Variants

This repository contains a PyTorch implementation of three Recurrent Neural Network (RNN) architectures trained from scratch to generate Indian full names (First Name + Last Name) at the character level. 

The goal of this project is to explore how different sequential architectures perform on character-level autoregressive generation, with a focus on evaluating generated name **Novelty** and **Diversity**.

The three models implemented are:
1. **Vanilla RNN** (Language Model)
2. **Bidirectional LSTM (BLSTM)** (Encoder-Decoder)
3. **AttentionRNN** (Self-Attention Language Model)

All recurrent cells (RNN, LSTM) and the attention mechanism (Bahdanau Additive Attention) are **implemented entirely from scratch** using basic PyTorch linear layers and parameters.

---

## Project Structure

| File | Description |
|---|---|
| `dataset.py` | Character vocabulary builder, dataset class, and data loading/padding utilities. |
| `models.py` | Implementations of the three model architectures and custom RNN/LSTM cells from scratch. |
| `train.py` | The training pipeline, including the training loop, optimizer setup, and name generation. |
| `evaluate.py` | Evaluation script that calculates **Novelty Rate** and **Diversity** metrics. |
| `output.md` | Comprehensive report detailing model architectures, parameter counts, and qualitative analysis. |
| `TrainingNames.txt` | Dataset of 1000 authentic Indian full names (first + last name) used for training. |
| `generated_*.txt` | Generated samples from the 3 trained models. |

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd nlu_problem2
   ```

2. **Install prerequisite libraries:**
   Requires Python 3.8+ and PyTorch.
   ```bash
   pip install torch matplotlib
   ```

---

## How to Run

### 1. Train the Models
To train the models from scratch and generate initial samples:

```bash
python train.py
```

This script will:
- Load and tokenize the dataset from `TrainingNames.txt`.
- Train the Vanilla RNN, BLSTM, and AttentionRNN for 100 epochs each.
- Generate 100 sample names from each model using temperature scaling (T=1.0).
- Save the trained model weights as `checkpoint_<ModelName>.pt`.
- Save generated samples to `generated_<ModelName>.txt`.
- Plot and save the overarching training loss curves to `loss_curves.png`.

### 2. Evaluate the Models
To calculate quantitative metrics (Novelty Rate and Diversity) and compare the models, run:

```bash
python evaluate.py
```

This will:
- Load the generated names.
- Calculate the quantitative metrics against the training dataset.
- Print an evaluation report to the console.
- Generate a comparison bar chart saved to `evaluation_comparison.png`.

---

## Quantitative Evaluation

Models were evaluated based on the generation of 100 names (temperature=1.0). 

| Model | Unique Generated | Novelty Rate (%) | Diversity |
|---|---|---|---|
| **Vanilla RNN** | 100 | **98.00%** | **1.0000** |
| **AttentionRNN** | 87 | **100.00%** | **0.8700** |
| **BLSTM** | 62 | **100.00%** | **0.6200** |

- **Novelty Rate**: Percentage of generated names NOT appearing in the training set. Higher = more creative generation.
- **Diversity**: Number of unique generated names divided by the total generated names. Higher = less repetitive.

---

## Qualitative Analysis & Samples

### 1. Vanilla RNN (Best Performance)
Generated highly realistic and indistinguishable Indian full names with perfectly learned capitalization and common suffix structures (-preet, -ika, -ansh) paired with authentic last names.
> **Samples:** *Aaravaj Naidu, Avuransh Thakur, Devayiya Pillai, Vivitika Agarwal, Manruansh Mehra*

### 2. AttentionRNN (Self-Attention)
Generates realistic names with proper capitalization. It maintains high diversity but occasionally attends too heavily to learned trailing suffixes like "-om" or "-preet".
> **Samples:** *Lakompreet Neshpreet, Vivom, Anomy, Nalul, Lakomna Ahuja, Lawompreet Bose*

### 3. Bidirectional LSTM (BLSTM)
Generates recognizable first and last name pairs but repeats components more often. It occasionally fails on capitalizing the first name.
> **Samples:** *anit Menon, nar Bose, Sanit Rastogi, anit Ganguly, Vina Goel*

---

## Key Insights & Architecture Takeaways

**Language Modeling vs Encoder-Decoder for Generation**

The key insight is that character-level name generation is fundamentally a **left-to-right sequential task**. Models structured as pure **language models** (like the Vanilla RNN and the self-attention AttentionRNN) perform drastically better because they directly learn `P(next_char | all_previous_chars)`. Their training procedure (next-character prediction) and iterative generation procedure (autoregressive sampling) are **perfectly aligned**.

Encoder-decoder architectures (like the bidirectional LSTM) suffer from **exposure bias** in this specific, uncontrolled generation task: during training, the encoder sees the entire full name via teacher forcing, but during test generation, it essentially receives nothing (or just a `<SOS>` token to start). This train-generation mismatch causes the decoder to operate in an out-of-distribution state, relying on fallback fragments and repeating sequences to generate text rather than naturally flowing names.
