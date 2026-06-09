"""Image eval: accuracy, ROC-AUC, and the false-positive rate (real flagged AI).

    python -m src.eval.eval_image
"""
from __future__ import annotations

import numpy as np

from src.config import CFG


def evaluate() -> dict:
    import pandas as pd
    from PIL import Image
    from sklearn.metrics import accuracy_score, roc_auc_score

    from src.models.image_detector import predict

    df = pd.read_parquet(CFG.image_manifest)
    df = df[df["split"] == "test"]

    probs, labels = [], []
    for _, row in df.iterrows():
        prob_ai = predict(Image.open(row["path"]))["ai_generated_prob"]
        probs.append(prob_ai)
        labels.append(int(row["label"]))

    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= 0.5).astype(int)
    neg = labels == 0
    fpr = float((preds[neg] == 1).mean()) if neg.any() else 0.0
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.5,
        "false_positive_rate": fpr,
    }


if __name__ == "__main__":
    print(evaluate())
