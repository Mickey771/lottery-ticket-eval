"""python -m lottery.plot results/imp_seed0.json results/imp_seed1.json"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("ticket", "winning ticket (IMP, original init)", "#2b6cb0"),
          ("oneshot", "one-shot pruning, original init", "#805ad5"),
          ("reinit", "IMP mask, random re-init", "#c05621"),
          ("random", "random mask, original init", "#718096")]


def main(paths, out="results/lottery.png"):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label, color in SERIES:
        for i, rows in enumerate(runs):
            rs = [r for r in rows if key in r]
            ax.plot([r["density"] * 100 for r in rs], [r[key] * 100 for r in rs], "o-", ms=3,
                    color=color, alpha=0.9 if i == 0 else 0.45, label=label if i == 0 else None)
    dense = sum(r[0]["ticket"] for r in runs) / len(runs) * 100
    ax.axhline(dense, color="black", ls=":", lw=1, label=f"dense network (mean {dense:.2f}%)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([100, 50, 20, 10, 5, 3])
    ax.set_xticklabels(["100", "50", "20", "10", "5", "3"])
    ax.set_xlabel("% of weights remaining")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"LeNet-300-100 on MNIST, {len(runs)} seeds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
