"""Reference: iterative magnitude pruning with rewinding to the original init.

Each round trains the current subnetwork from the original initialization,
removes 20% of the surviving hidden-layer weights and 10% of the output
layer's by magnitude, and rewinds. Stops once density is at most the target.
"""
from lottery.core import density, magnitude_prune, make_model

TARGET_DENSITY = 0.03
RATES = {"fc1": 0.2, "fc2": 0.2, "fc3": 0.1}


def find_mask(init, data, train_fn, device):
    masks = make_model(0).masks()
    masks = {n: m.to(device) for n, m in masks.items()}
    while density(masks) > TARGET_DENSITY:
        model = make_model(0).to(device)
        model.load_state_dict(init, strict=False)
        model.set_masks(masks)
        train_fn(model, data)
        masks = magnitude_prune(model, RATES)
    return masks
