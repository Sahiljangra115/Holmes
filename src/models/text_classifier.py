"""Fine-tune DeBERTa-v3-base on the binary LIAR2 target, calibrate with
temperature scaling, log to MLflow, and expose predict().

Training needs a GPU and the prepared parquet files. predict() and the
calibration math are CPU-friendly and are what the API and tests exercise.

    python -m src.models.text_classifier        # trains
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.config import CFG


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(temp: float, logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits / max(temp, 1e-3))
    p = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Golden-section-free 1D search over T in [0.25, 5.0]. Small and robust."""
    grid = np.linspace(0.25, 5.0, 96)
    losses = [_nll(t, logits, labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def save_temperature(temp: float) -> None:
    CFG.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    CFG.calibration_path.write_text(json.dumps({"temperature": temp}))


def load_temperature() -> float:
    if CFG.calibration_path.exists():
        return float(json.loads(CFG.calibration_path.read_text())["temperature"])
    return 1.0


@dataclass
class TextPrediction:
    label: str
    label_id: int
    confidence: float


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train():
    import datasets
    import mlflow
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(CFG.text_model_name)
    ds = datasets.load_dataset(
        "parquet",
        data_files={
            "train": str(CFG.train_parquet),
            "val": str(CFG.val_parquet),
            "test": str(CFG.test_parquet),
        },
    )

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=CFG.text_max_len)

    ds = ds.map(encode, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.text_model_name, num_labels=CFG.text_num_labels
    )

    args = TrainingArguments(
        output_dir=str(CFG.text_model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        learning_rate=CFG.text_lr,
        per_device_train_batch_size=CFG.text_batch_size,
        per_device_eval_batch_size=CFG.text_batch_size,
        num_train_epochs=CFG.text_epochs,
        seed=CFG.seed,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics,
    )

    mlflow.set_tracking_uri("file:mlruns")
    with mlflow.start_run(run_name="deberta-liar2-binary"):
        mlflow.log_params(
            {
                "model": CFG.text_model_name,
                "lr": CFG.text_lr,
                "epochs": CFG.text_epochs,
                "batch": CFG.text_batch_size,
                "seed": CFG.seed,
                "max_len": CFG.text_max_len,
            }
        )
        trainer.train()
        val_metrics = trainer.evaluate(ds["val"])
        mlflow.log_metrics({k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))})

        val_logits = trainer.predict(ds["val"]).predictions
        val_labels = np.array(ds["val"]["label"])
        temp = fit_temperature(val_logits, val_labels)
        save_temperature(temp)
        mlflow.log_metric("temperature", temp)

    trainer.save_model(str(CFG.text_model_dir))
    tok.save_pretrained(str(CFG.text_model_dir))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_BUNDLE = None


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(CFG.text_model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(CFG.text_model_dir))
        model.eval()
        _BUNDLE = (tok, model, load_temperature(), torch)
    return _BUNDLE


def predict(text: str) -> TextPrediction:
    tok, model, temp, torch = _load()
    enc = tok(text, truncation=True, max_length=CFG.text_max_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    probs = _softmax(logits / temp)[0]
    label_id = int(probs.argmax())
    return TextPrediction(
        label=CFG.text_label_names[label_id],
        label_id=label_id,
        confidence=float(probs[label_id]),
    )


if __name__ == "__main__":
    train()
