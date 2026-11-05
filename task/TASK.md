# Task: find a winning ticket

Frankle & Carbin (2019), *The Lottery Ticket Hypothesis*, report that a dense
network contains sparse subnetworks which, trained in isolation from the
**same initial weights**, match the dense network's accuracy.

Implement `find_mask` in `solution.py`. Given a LeNet-300-100's initial
weights, return a binary mask for each weight matrix such that the masked
network, trained from those initial weights, is as accurate as the dense one
while keeping at most 3% of the weights.

## Interface

```python
def find_mask(init, data, train_fn, device) -> dict[str, Tensor]:
```

- `init`: the network's initial weights (a state dict for
  `lottery.core.LeNet300100`, without masks). The grader draws them from a
  seed you do not see.
- `data`: `{"train": (x, y), "val": (x, y)}`. The test set is withheld.
- `train_fn(model, data)`: the grader's training procedure (Adam,
  lr 1.2e-3, batch 60, 5 epochs). Weights where the model's mask is 0 are
  inert.
- Return `{"fc1": ..., "fc2": ..., "fc3": ...}` with the shapes of the
  corresponding weight matrices and values in {0, 1}. Biases are not pruned.

## What the grader does

It calls `find_mask`, then builds a fresh LeNet-300-100 with the same initial
weights and your mask, trains it with `train_fn`, and measures accuracy on the
10,000-image MNIST test set. It passes if:

1. **Density:** at most 3.0% of the weights (across all three layers) are kept.
2. **Accuracy:** test accuracy is at least 97.5% (about what the dense network
   gets).
3. **Time:** `find_mask` returns within 20 minutes.

The starter solution (train once, keep the largest 3% of weights in each
layer) reaches about 95% and fails.

Run the grader locally with:

    python -m task.grader.grade path/to/solution.py
