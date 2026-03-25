"""
train.py — Training Script for Character-Level Name Generation Models
=====================================================================

This script trains three models sequentially:
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Bahdanau Attention

For each model it:
  - Trains for a fixed number of epochs using cross-entropy loss
  - Saves a checkpoint (.pt file)
  - Generates sample names via temperature-scaled sampling
  - Records per-epoch loss for plotting

Key improvements over baseline:
  - Dropout = 0.2 (stronger regularisation → less memorisation → higher novelty)
  - Temperature = 1.0 for generation (more diverse/novel outputs)
  - Gradient clipping to prevent exploding gradients

Author : Auto-generated for NLU Problem 2
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from dataset import load_names, CharVocab, NameDataset, collate_fn
from models import VanillaRNN, BLSTMGenerator, AttentionRNN


# ═══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
EMBED_SIZE   = 64       # character embedding dimension
HIDDEN_SIZE  = 128      # RNN hidden state dimension
NUM_LAYERS   = 2        # number of stacked recurrent layers
DROPOUT      = 0.2      # dropout rate (higher → more regularisation)
LR           = 0.003    # Adam learning rate
BATCH_SIZE   = 64       # mini-batch size
EPOCHS       = 100      # number of training epochs
GRAD_CLIP    = 5.0      # maximum gradient norm for clipping
TEMPERATURE  = 1.0      # sampling temperature (1.0 = balanced novelty/quality)
NUM_GENERATE = 100      # how many names to generate per model
DATA_PATH    = "TrainingNames.txt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store hyperparameters in a dict for JSON serialisation
HYPERPARAMS = {
    "embed_size":    EMBED_SIZE,
    "hidden_size":   HIDDEN_SIZE,
    "num_layers":    NUM_LAYERS,
    "dropout":       DROPOUT,
    "learning_rate": LR,
    "batch_size":    BATCH_SIZE,
    "epochs":        EPOCHS,
    "grad_clip":     GRAD_CLIP,
    "temperature":   TEMPERATURE,
    "device":        str(DEVICE),
}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP  (one epoch)
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimiser, criterion, pad_idx, device):
    """
    Train the model for one epoch over all batches.

    Uses next-character prediction loss:
      input  = [SOS, c1, c2, …, c_{n-1}]
      target = [c1,  c2, c3, …, EOS     ]

    Parameters
    ----------
    model     : nn.Module        — the model to train
    loader    : DataLoader       — batched training data
    optimiser : torch.optim      — Adam (or similar) optimiser
    criterion : nn.CrossEntropyLoss (with ignore_index=pad_idx)
    pad_idx   : int              — <PAD> token index to ignore in loss
    device    : torch.device

    Returns
    -------
    float — average per-token loss over the epoch
    """
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for batch in loader:
        batch = batch.to(device)

        # Split into input (drop last) and target (drop first)
        inp = batch[:, :-1]    # SOS, c1, … , c_{n-1}
        tgt = batch[:, 1:]     # c1,  c2, … , EOS

        # Forward pass
        logits, _ = model(inp)

        # Flatten for cross-entropy:  (B*T, V) vs (B*T,)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat    = tgt.reshape(-1)

        loss = criterion(logits_flat, tgt_flat)

        # Backward pass with gradient clipping
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()

        # Accumulate loss (only for non-padding tokens)
        mask = (tgt_flat != pad_idx)
        total_loss   += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    return total_loss / max(total_tokens, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  NAME GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_names(model, model_name, vocab, device,
                   n=NUM_GENERATE, temperature=TEMPERATURE, max_len=30):
    """
    Generate `n` names from a trained model.

    For encoder-decoder models (BLSTM, AttentionRNN) uses the model's
    .generate() method.  For the VanillaRNN language model, performs
    autoregressive sampling manually.

    Parameters
    ----------
    model       : nn.Module
    model_name  : str          — for logging
    vocab       : CharVocab
    device      : torch.device
    n           : int          — number of names to generate
    temperature : float        — sampling temperature
    max_len     : int          — maximum characters per name

    Returns
    -------
    list[str] — generated name strings (non-empty only)
    """
    model.eval()
    names = []

    with torch.no_grad():
        for _ in range(n):
            if hasattr(model, "generate"):
                # ── Encoder-decoder models have a dedicated generate method ──
                name = model.generate(
                    vocab.sos_idx, vocab.eos_idx, vocab, device,
                    max_len=max_len, temperature=temperature
                )
            else:
                # ── Vanilla RNN: manual autoregressive sampling ──────────────
                hidden = None
                token  = torch.tensor([[vocab.sos_idx]], device=device)  # (1, 1)
                result = []

                for __ in range(max_len):
                    logits, hidden = model(token, hidden)
                    # Scale logits by temperature before softmax
                    logits = logits[:, -1, :] / temperature
                    probs  = torch.softmax(logits, dim=-1)
                    next_tok = torch.multinomial(probs, 1)  # (1, 1)

                    if next_tok.item() == vocab.eos_idx:
                        break
                    result.append(next_tok.item())
                    token = next_tok   # feed sampled token as next input

                name = vocab.decode(result)

            if name:   # skip any empty strings
                names.append(name)

    return names


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def save_loss_plot(all_losses: dict, path: str = "loss_curves.png"):
    """
    Save a line plot of training losses for all models.

    Parameters
    ----------
    all_losses : dict[str, list[float]]  — model_name → per-epoch losses
    path       : str                     — output file path
    """
    plt.figure(figsize=(10, 6))
    for name, losses in all_losses.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (per token)")
    plt.title("Training Loss Curves — Character-Level Name Generation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Loss plot saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main training pipeline: load data → train models → generate → save."""

    print(f"Device: {DEVICE}")
    print(f"Loading names from {DATA_PATH} ...")
    names = load_names(DATA_PATH)
    print(f"  Loaded {len(names)} names")

    # ── Build vocabulary and dataset ─────────────────────────────────────
    vocab = CharVocab(names)
    print(f"  Vocab size: {len(vocab)} ({len(vocab) - 3} unique chars + 3 special tokens)")

    dataset = NameDataset(names, vocab)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx),
    )

    # Cross-entropy loss ignoring <PAD> tokens
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # ── Define all three models ──────────────────────────────────────────
    models_cfg = {
        "VanillaRNN":   VanillaRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
        "BLSTM":        BLSTMGenerator(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
        "AttentionRNN": AttentionRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT),
    }

    all_losses = {}   # model_name → list of per-epoch losses
    results    = {}   # model_name → summary dict

    # ── Train each model sequentially ────────────────────────────────────
    for model_name, model in models_cfg.items():
        model = model.to(DEVICE)
        param_count = model.count_parameters()

        print(f"\n{'='*60}")
        print(f"Training {model_name}  ({param_count:,} trainable parameters)")
        print(f"{'='*60}")

        optimiser = torch.optim.Adam(model.parameters(), lr=LR)
        losses = []
        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, loader, optimiser, criterion,
                                   vocab.pad_idx, DEVICE)
            losses.append(loss)

            # Print progress every 10 epochs (and the first epoch)
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  [{elapsed:.1f}s]")

        all_losses[model_name] = losses

        # ── Save model checkpoint ────────────────────────────────────────
        ckpt_path = f"checkpoint_{model_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

        # ── Generate sample names ────────────────────────────────────────
        gen_names = generate_names(model, model_name, vocab, DEVICE)
        gen_path  = f"generated_{model_name}.txt"
        with open(gen_path, "w") as f:
            f.write("\n".join(gen_names))
        print(f"  Generated {len(gen_names)} names → {gen_path}")

        # ── Record results ───────────────────────────────────────────────
        results[model_name] = {
            "param_count":     param_count,
            "final_loss":      losses[-1],
            "generated_count": len(gen_names),
        }

    # ── Save loss plot ───────────────────────────────────────────────────
    save_loss_plot(all_losses)

    # ── Save training summary as JSON ────────────────────────────────────
    summary = {"hyperparameters": HYPERPARAMS, "models": results}
    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved → training_summary.json")
    print("Done!")


if __name__ == "__main__":
    main()
