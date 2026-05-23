"""TF-IDF + Logistic Regression baseline.

This is the honesty anchor. The transformer must beat this on the same test set
or the transformer is not earning its keep. Kept deliberately simple.

    python -m src.models.baseline_text
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.config import CFG


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_baseline() -> Pipeline:
    train = pd.read_parquet(CFG.train_parquet)
    val = pd.read_parquet(CFG.val_parquet)

    pipe = build_pipeline()
    pipe.fit(train["text"], train["label"])

    preds = pipe.predict(val["text"])
    macro = f1_score(val["label"], preds, average="macro")
    weighted = f1_score(val["label"], preds, average="weighted")
    print(classification_report(val["label"], preds, target_names=list(CFG.text_label_names)))
    print(f"baseline val macro-F1={macro:.4f} weighted-F1={weighted:.4f}")

    CFG.text_model_dir.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, CFG.text_model_dir.parent / "baseline_text.joblib")
    return pipe


if __name__ == "__main__":
    train_baseline()
