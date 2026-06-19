"""Temperature scaling: a confident-but-wrong model should get T > 1 (cooled),
and the ECE helper should drop after calibration."""
import numpy as np

from src.eval.eval_text import expected_calibration_error
from src.models.text_classifier import _softmax, fit_temperature


def test_temperature_cools_overconfident_logits():
    rng = np.random.default_rng(0)
    n = 400
    labels = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.7
    logits = np.zeros((n, 2))
    for i in range(n):
        win = labels[i] if correct[i] else 1 - labels[i]
        logits[i, win] = 6.0
    t = fit_temperature(logits, labels)
    assert t > 1.0


def test_ece_improves_after_calibration():
    rng = np.random.default_rng(1)
    n = 400
    labels = rng.integers(0, 2, n)
    correct = rng.random(n) < 0.7
    logits = np.zeros((n, 2))
    for i in range(n):
        win = labels[i] if correct[i] else 1 - labels[i]
        logits[i, win] = 6.0

    raw = _softmax(logits)
    t = fit_temperature(logits, labels)
    cal = _softmax(logits / t)
    pred = raw.argmax(1)
    ece_raw = expected_calibration_error(raw.max(1), pred == labels)
    ece_cal = expected_calibration_error(cal.max(1), pred == labels)
    assert ece_cal <= ece_raw
