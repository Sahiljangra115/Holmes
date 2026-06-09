"""Text eval: per-class precision/recall/F1, confusion matrix, calibration curve.

    python -m src.eval.eval_text
"""
from __future__ import annotations

import numpy as np

from src.config import CFG


def evaluate() -> dict:
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    from src.models.text_classifier import predict

    test = pd.read_parquet(CFG.test_parquet)
    preds, confs = [], []
    for text in test["text"]:
        p = predict(text)
        preds.append(p.label_id)
        confs.append(p.confidence)

    y = test["label"].to_numpy()
    report = classification_report(y, preds, target_names=list(CFG.text_label_names), output_dict=True)
    cm = confusion_matrix(y, preds).tolist()
    ece = expected_calibration_error(np.array(confs), np.array(preds) == y)
    return {"report": report, "confusion_matrix": cm, "ece": ece}


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Are the confidences honest? Bin by confidence, compare mean confidence to
    accuracy in each bin. Lower is better."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


if __name__ == "__main__":
    print(evaluate())
