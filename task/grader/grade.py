"""Grade a lottery-ticket solution.

    python -m task.grader.grade path/to/solution.py [--seed N]

Exit 0 = pass, 1 = fail. The grader owns the initialization (hidden seed),
the data split, and the final training run; the solution supplies only the
mask.
"""
import argparse
import importlib.util
import json
import sys
import time

import torch

from lottery.core import LAYERS, accuracy, density, load_mnist, make_model, train, weight_state

HIDDEN_SEED = 7
MAX_DENSITY = 0.03
MIN_TEST_ACC = 0.975
TIME_LIMIT_S = 20 * 60
EPOCHS = 5


def load(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_fn(model, data, epochs=EPOCHS):
    """The only training procedure solutions should use (same as the final run)."""
    return train(model, data, epochs=epochs)


def check_masks(masks, ref):
    if set(masks) != set(LAYERS):
        return f"masks must have keys {list(LAYERS)}, got {sorted(masks)}"
    for n in LAYERS:
        m = masks[n]
        if tuple(m.shape) != tuple(ref[n].shape):
            return f"mask {n} has shape {tuple(m.shape)}, expected {tuple(ref[n].shape)}"
        if not torch.all((m == 0) | (m == 1)):
            return f"mask {n} is not binary"
    return None


def grade(path, seed=HIDDEN_SEED, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    sol = load(path)
    data = load_mnist(device)
    visible = {"train": data["train"], "val": data["val"]}  # no test set
    init = weight_state(make_model(seed))

    t0 = time.time()
    masks = sol.find_mask({k: v.clone() for k, v in init.items()}, visible, train_fn, device)
    elapsed = time.time() - t0
    masks = {n: torch.as_tensor(m).float().to(device) for n, m in masks.items()}
    err = check_masks(masks, make_model(seed).masks())
    if err:
        return {"pass": False, "reason": err}

    model = make_model(seed).to(device)
    model.load_state_dict({k: v.to(device) for k, v in init.items()}, strict=False)
    model.set_masks(masks)
    train(model, data, epochs=EPOCHS)
    acc = accuracy(model, *data["test"])
    dens = density(masks)
    checks = {
        "density": dens <= MAX_DENSITY,
        "test_accuracy": acc >= MIN_TEST_ACC,
        "time_limit": elapsed <= TIME_LIMIT_S,
    }
    return {"pass": all(checks.values()), "checks": checks, "density": dens,
            "test_acc": acc, "search_seconds": round(elapsed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--seed", type=int, default=HIDDEN_SEED)
    a = ap.parse_args()
    result = grade(a.solution, seed=a.seed)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
