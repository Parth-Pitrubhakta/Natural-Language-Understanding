"""
evaluate.py — Quantitative Evaluation for Character-Level Name Generation
=========================================================================

Computes two metrics for each model's generated names:

  1. **Novelty Rate** — percentage of generated names that do NOT appear
     in the training set.  Higher = more creative.

  2. **Diversity** — number of unique generated names divided by total
     generated names.  Higher = less repetitive.

Also produces:
  - A comparison bar chart (PNG)
  - A JSON file with all evaluation results

Author : Auto-generated for NLU Problem 2
"""

import os
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from dataset import load_names


# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAMES = ["VanillaRNN", "BLSTM", "AttentionRNN"]
DATA_PATH   = "TrainingNames.txt"


def compute_metrics(generated: list[str], training_set: set[str]):
    """
    Compute novelty rate and diversity for a list of generated names.

    Parameters
    ----------
    generated    : list[str]  — names produced by the model
    training_set : set[str]   — the training names (for novelty check)

    Returns
    -------
    novelty_rate : float  — percentage of novel names (0–100)
    diversity    : float  — fraction of unique names (0–1)
    """
    total = len(generated)
    if total == 0:
        return 0.0, 0.0

    # Count names that are NOT in the training set
    novel  = sum(1 for n in generated if n not in training_set)
    # Count distinct names
    unique = len(set(generated))

    novelty_rate = (novel / total) * 100
    diversity    = unique / total
    return novelty_rate, diversity


def main():
    """Load generated names, compute metrics, print table, save chart & JSON."""

    # ── Load the training set for comparison ─────────────────────────────
    training_names = load_names(DATA_PATH)
    training_set   = set(n.strip() for n in training_names)
    print(f"Training set size: {len(training_set)} unique names\n")

    # ── Load training summary (for parameter counts) ────────────────────
    summary = {}
    if os.path.exists("training_summary.json"):
        with open("training_summary.json") as f:
            summary = json.load(f)

    results     = {}    # model → metrics dict
    all_samples = {}    # model → generated names list

    # ── Evaluate each model ──────────────────────────────────────────────
    for model_name in MODEL_NAMES:
        gen_path = f"generated_{model_name}.txt"
        if not os.path.exists(gen_path):
            print(f"  [SKIP] {gen_path} not found")
            continue

        # Read generated names
        with open(gen_path) as f:
            generated = [line.strip() for line in f if line.strip()]

        # Compute metrics
        novelty, diversity = compute_metrics(generated, training_set)
        unique_count = len(set(generated))
        total        = len(generated)

        # Store results
        results[model_name] = {
            "total_generated":  total,
            "unique_generated": unique_count,
            "novelty_rate":     round(novelty, 2),
            "diversity":        round(diversity, 4),
        }
        all_samples[model_name] = generated

        # ── Look up parameter count from training summary ────────────────
        param_count = "N/A"
        if summary and "models" in summary and model_name in summary["models"]:
            param_count = f"{summary['models'][model_name]['param_count']:,}"

        # ── Print per-model report ───────────────────────────────────────
        print(f"─── {model_name} (params: {param_count}) ───")
        print(f"  Total generated : {total}")
        print(f"  Unique generated: {unique_count}")
        print(f"  Novelty Rate    : {novelty:.2f}%")
        print(f"  Diversity       : {diversity:.4f}")
        print(f"  Sample names    : {', '.join(generated[:10])}")

        # Show which generated names also appear in the training set
        in_train = [n for n in generated if n in training_set]
        print(f"  In training set : {len(in_train)} names")
        if in_train:
            print(f"    Examples      : {', '.join(in_train[:5])}")
        print()

    # ═════════════════════════════════════════════════════════════════════
    #  COMPARISON BAR CHART
    # ═════════════════════════════════════════════════════════════════════
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        names       = list(results.keys())
        novelties   = [results[n]["novelty_rate"]  for n in names]
        diversities = [results[n]["diversity"]     for n in names]

        # Colour palette
        colors = ["#4e79a7", "#f28e2b", "#e15759"]

        # ── Novelty Rate subplot ─────────────────────────────────────────
        axes[0].bar(names, novelties, color=colors[:len(names)])
        axes[0].set_ylabel("Novelty Rate (%)")
        axes[0].set_title("Novelty Rate Comparison")
        axes[0].set_ylim(0, 105)
        for i, v in enumerate(novelties):
            axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

        # ── Diversity subplot ────────────────────────────────────────────
        axes[1].bar(names, diversities, color=colors[:len(names)])
        axes[1].set_ylabel("Diversity")
        axes[1].set_title("Diversity Comparison")
        axes[1].set_ylim(0, 1.1)
        for i, v in enumerate(diversities):
            axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig("evaluation_comparison.png", dpi=150)
        plt.close()
        print("Evaluation chart saved → evaluation_comparison.png")

    # ── Save results to JSON ─────────────────────────────────────────────
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Evaluation results saved → evaluation_results.json")


if __name__ == "__main__":
    main()
