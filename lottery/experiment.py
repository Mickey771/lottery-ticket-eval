"""Iterative magnitude pruning (IMP) with rewinding, plus controls.

    python -m lottery.experiment --seed 0 --rounds 16

At each sparsity level reached by IMP, trains and tests:
  ticket    IMP mask, weights reset to the original init (the winning ticket)
  reinit    IMP mask, fresh random init
  random    random mask with the same per-layer counts, original init
  oneshot   mask from pruning the dense trained net straight to that level,
            original init
"""
import argparse
import json
from pathlib import Path

from lottery.core import (LAYERS, Timer, accuracy, density, load_mnist, magnitude_prune,
                          make_model, random_masks_like, train, weight_state)

RATES = {"fc1": 0.2, "fc2": 0.2, "fc3": 0.1}  # per round, as in the paper


def train_from(init, masks, data, seed, epochs, device):
    m = make_model(seed).to(device)
    if init is not None:
        m.load_state_dict({**init, **{f"{n}_mask": masks[n] for n in LAYERS}})
    m.set_masks(masks)
    train(m, data, epochs=epochs)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    data = load_mnist(a.device)
    xte, yte = data["test"]
    t = Timer()
    model = make_model(a.seed).to(a.device)
    init = weight_state(model)
    dense_trained = None
    rows = []
    for r in range(a.rounds + 1):
        masks = model.masks()
        dens = density(masks)
        ticket = train_from(init, masks, data, a.seed, a.epochs, a.device)
        if r == 0:
            dense_trained = ticket
        row = {"round": r, "density": dens, "ticket": accuracy(ticket, xte, yte)}
        if r > 0:
            reinit = train_from(None, masks, data, 10_000 + a.seed * 100 + r, a.epochs, a.device)
            row["reinit"] = accuracy(reinit, xte, yte)
            rnd = random_masks_like(masks, seed=20_000 + a.seed * 100 + r)
            row["random"] = accuracy(train_from(init, rnd, data, a.seed, a.epochs, a.device), xte, yte)
            # one-shot: prune the dense trained net once to this round's per-layer counts
            frac_removed = {n: 1 - masks[n].mean().item() for n in LAYERS}
            dense_trained.set_masks({n: dense_trained.masks()[n].fill_(1) for n in LAYERS})
            one = magnitude_prune(dense_trained, frac_removed)
            row["oneshot"] = accuracy(train_from(init, one, data, a.seed, a.epochs, a.device), xte, yte)
        rows.append(row)
        print(f"[{t}] round {r:2d} density {dens * 100:6.2f}%  " +
              "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k not in ("round", "density")),
              flush=True)
        # prune the trained ticket, then rewind for the next round
        model = make_model(a.seed).to(a.device)
        model.load_state_dict(init, strict=False)
        model.set_masks(magnitude_prune(ticket, RATES))

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / f"imp_seed{a.seed}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
