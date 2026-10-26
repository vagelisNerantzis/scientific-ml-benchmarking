"""Top-label equal-width ECE, with explicit probability validation."""
import numpy as np

def normalize_probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or not p.shape[0] or p.shape[1] < 2:
        raise ValueError('Expected nonempty N by K probabilities, K >= 2')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-6).any():
        raise ValueError('Probabilities must be finite and between zero and one')
    sums = p.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1, atol=1e-5, rtol=0):
        raise ValueError('Probability rows must sum to one')
    return p / sums

def expected_calibration_error(probabilities, y_true, *, labels, bins=10):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('bins must be a positive integer')
    p = normalize_probabilities(probabilities)
    labels = list(labels)
    if len(labels) != p.shape[1] or len(set(labels)) != len(labels):
        raise ValueError('Probability columns require a unique ordered class schema')
    y = np.asarray(y_true)
    if y.ndim != 1 or len(y) != len(p) or not set(y).issubset(labels):
        raise ValueError('Invalid labels or row count')
    confidence = p.max(axis=1)
    correct = np.asarray(labels)[p.argmax(axis=1)] == y
    result = 0.0
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        mask = (confidence >= lower if i == 0 else confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)
