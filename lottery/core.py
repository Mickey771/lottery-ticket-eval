"""MNIST, LeNet-300-100, masked training and magnitude pruning.

Setup follows Frankle & Carbin (2019), "The Lottery Ticket Hypothesis":
a 784-300-100-10 fully connected network trained with Adam. Pruning removes
the smallest-magnitude surviving weights layer by layer; a "winning ticket"
is the pruned network reset to its original initialization.
"""
import gzip
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LAYERS = ("fc1", "fc2", "fc3")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(device="cpu"):
    """Return dict of (x, y) tensors: 'train' (55k), 'val' (5k), 'test' (10k)."""
    def xs(name):
        x = torch.tensor(_read_idx(DATA_DIR / f"{name}-images-idx3-ubyte.gz"), dtype=torch.float32)
        return ((x / 255.0 - 0.1307) / 0.3081).reshape(len(x), -1).to(device)

    def ys(name):
        return torch.tensor(_read_idx(DATA_DIR / f"{name}-labels-idx1-ubyte.gz"), dtype=torch.long).to(device)

    xtr, ytr = xs("train"), ys("train")
    return {
        "train": (xtr[:55000], ytr[:55000]),
        "val": (xtr[55000:], ytr[55000:]),
        "test": (xs("t10k"), ys("t10k")),
    }


class LeNet300100(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        for name in LAYERS:
            self.register_buffer(f"{name}_mask", torch.ones_like(getattr(self, name).weight))

    def masks(self):
        return {n: getattr(self, f"{n}_mask") for n in LAYERS}

    def set_masks(self, masks):
        for n in LAYERS:
            getattr(self, f"{n}_mask").copy_(masks[n])

    def _lin(self, name, x):
        layer = getattr(self, name)
        return F.linear(x, layer.weight * getattr(self, f"{name}_mask"), layer.bias)

    def forward(self, x):
        x = F.relu(self._lin("fc1", x))
        x = F.relu(self._lin("fc2", x))
        return self._lin("fc3", x)


def make_model(seed):
    torch.manual_seed(seed)
    return LeNet300100()


def weight_state(model):
    """Copy of the trainable weights (not the masks)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if not k.endswith("_mask")}


def density(masks):
    kept = sum(m.sum().item() for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    return kept / total


@torch.no_grad()
def accuracy(model, x, y, batch=10000):
    model.eval()
    correct = sum((model(x[i:i + batch]).argmax(-1) == y[i:i + batch]).sum().item()
                  for i in range(0, len(x), batch))
    model.train()
    return correct / len(x)


def train(model, data, epochs=5, lr=1.2e-3, batch_size=60, data_seed=0):
    """Train in place with Adam on data['train']; pruned weights stay inert."""
    x, y = data["train"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(data_seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g).to(x.device)
        for i in range(0, len(x), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def magnitude_prune(model, rates):
    """New masks removing `rates[layer]` of each layer's surviving weights by |w|."""
    new = {}
    for n in LAYERS:
        w = getattr(model, n).weight.detach().abs()
        m = getattr(model, f"{n}_mask")
        alive = w[m.bool()]
        k = int(round(rates[n] * alive.numel()))
        if k == 0:
            new[n] = m.clone()
            continue
        thresh = alive.kthvalue(k).values
        new[n] = (m.bool() & (w > thresh)).float()
    return new


def random_masks_like(masks, seed):
    """Random masks with the same per-layer counts as `masks`."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, m in masks.items():
        k = int(m.sum().item())
        flat = torch.zeros(m.numel())
        flat[torch.randperm(m.numel(), generator=g)[:k]] = 1
        out[n] = flat.reshape(m.shape).to(m.device)
    return out


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __str__(self):
        return f"{time.time() - self.t0:.0f}s"
