"""
Quick script to generate names from saved BLSTM checkpoint and then train AttentionRNN.
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import load_names, CharVocab, NameDataset, collate_fn
from models import VanillaRNN, BLSTMGenerator, AttentionRNN
from train import (EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LR, BATCH_SIZE,
                   EPOCHS, GRAD_CLIP, TEMPERATURE, NUM_GENERATE, HYPERPARAMS,
                   train_one_epoch, generate_names, save_loss_plot)

DATA_PATH = "TrainingNames.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    names = load_names(DATA_PATH)
    vocab = CharVocab(names)
    dataset = NameDataset(names, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, vocab.pad_idx))
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    # ── Load existing training summary or create new one ──
    if os.path.exists("training_summary.json"):
        with open("training_summary.json") as f:
            summary = json.load(f)
    else:
        summary = {"hyperparameters": HYPERPARAMS, "models": {}}
    all_losses = {}

    # ── Step 1: Generate BLSTM names from checkpoint ──
    print("Loading BLSTM checkpoint and generating names...")
    blstm = BLSTMGenerator(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    blstm.load_state_dict(torch.load("checkpoint_BLSTM.pt", map_location=DEVICE, weights_only=True))
    gen_names_blstm = generate_names(blstm, "BLSTM", vocab, DEVICE)
    with open("generated_BLSTM.txt", "w") as f:
        f.write("\n".join(gen_names_blstm))
    print(f"  BLSTM: Generated {len(gen_names_blstm)} names")
    if "BLSTM" not in summary["models"]:
        summary["models"]["BLSTM"] = {"param_count": blstm.count_parameters(), "final_loss": None}
    summary["models"]["BLSTM"]["generated_count"] = len(gen_names_blstm)

    # ── Step 2: Train AttentionRNN ──
    print(f"\n{'='*60}")
    attn = AttentionRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    param_count = attn.count_parameters()
    print(f"Training AttentionRNN  ({param_count:,} trainable parameters)")
    print(f"{'='*60}")

    optimiser = torch.optim.Adam(attn.parameters(), lr=LR)
    losses = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(attn, loader, optimiser, criterion, vocab.pad_idx, DEVICE)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  [{elapsed:.1f}s]")

    all_losses["AttentionRNN"] = losses
    torch.save(attn.state_dict(), "checkpoint_AttentionRNN.pt")
    print("  Checkpoint saved → checkpoint_AttentionRNN.pt")

    gen_names_attn = generate_names(attn, "AttentionRNN", vocab, DEVICE)
    with open("generated_AttentionRNN.txt", "w") as f:
        f.write("\n".join(gen_names_attn))
    print(f"  Generated {len(gen_names_attn)} names → generated_AttentionRNN.txt")

    summary["models"]["AttentionRNN"] = {
        "param_count": param_count,
        "final_loss": losses[-1],
        "generated_count": len(gen_names_attn),
    }

    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Updated training_summary.json")
    print("Done!")


if __name__ == "__main__":
    main()
