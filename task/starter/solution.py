"""Starter: one-shot magnitude pruning to the target density."""
from lottery.core import LAYERS, magnitude_prune, make_model

TARGET_DENSITY = 0.03


def find_mask(init, data, train_fn, device):
    model = make_model(0).to(device)
    model.load_state_dict(init, strict=False)
    train_fn(model, data)
    return magnitude_prune(model, {n: 1 - TARGET_DENSITY for n in LAYERS})
