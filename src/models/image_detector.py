"""EfficientNet-B0 real-vs-AI detector: fine-tune, pick best by val AUC, log
accuracy and false-positive rate to MLflow, export ONNX, run inference.

Why best-by-AUC and not accuracy: the cost of flagging a real photo as AI (a
false positive) is high, and the set can be imbalanced. AUC and FPR tell the
honest story that accuracy hides.

    python -m src.models.image_detector        # trains (needs GPU + manifest)
"""
from __future__ import annotations

import numpy as np

from src.config import CFG


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _build_loaders():
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    df = pd.read_parquet(CFG.image_manifest)

    train_tf = transforms.Compose(
        [
            transforms.Resize((CFG.image_size, CFG.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(CFG.image_size, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.JPEG(quality=(60, 95))], p=0.5)
            if hasattr(transforms, "JPEG")
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize((CFG.image_size, CFG.image_size)), transforms.ToTensor()]
    )

    class ImgDS(Dataset):
        def __init__(self, frame, tf):
            self.frame = frame.reset_index(drop=True)
            self.tf = tf

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            row = self.frame.iloc[i]
            img = Image.open(row["path"]).convert("RGB")
            return self.tf(img), int(row["label"])

    def loader(split, tf, shuffle):
        return DataLoader(
            ImgDS(df[df["split"] == split], tf),
            batch_size=CFG.image_batch_size,
            shuffle=shuffle,
            num_workers=2,
        )

    return loader("train", train_tf, True), loader("val", eval_tf, False)


def build_backbone():
    import timm

    return timm.create_model(CFG.image_model_name, num_classes=2, pretrained=True)


def _val_auc_fpr(model, val_loader, device) -> tuple[float, float]:
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y.tolist())
    probs, labels = np.array(probs), np.array(labels)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
    preds = (probs >= 0.5).astype(int)
    neg = labels == 0
    fpr = float((preds[neg] == 1).mean()) if neg.any() else 0.0
    return float(auc), fpr


def train():
    import mlflow
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = _build_loaders()
    model = build_backbone().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.image_lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    mlflow.set_tracking_uri("file:mlruns")
    best_auc, best_state = -1.0, None
    with mlflow.start_run(run_name="efficientnet-real-vs-ai"):
        mlflow.log_params({"model": CFG.image_model_name, "lr": CFG.image_lr, "epochs": CFG.image_epochs})
        for epoch in range(CFG.image_epochs):
            for p in model.parameters():
                p.requires_grad = epoch > 0 or True
            model.train()
            for x, y in train_loader:
                opt.zero_grad()
                loss = loss_fn(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
            auc, fpr = _val_auc_fpr(model, val_loader, device)
            mlflow.log_metrics({"val_auc": auc, "val_fpr": fpr}, step=epoch)
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    export_onnx(model)
    return best_auc


def export_onnx(model, path=None):
    import torch

    path = str(path or CFG.image_onnx_path)
    CFG.image_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, 3, CFG.image_size, CFG.image_size)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(str(CFG.image_onnx_path), providers=["CPUExecutionProvider"])
    return _SESSION


def _preprocess(image) -> np.ndarray:
    from PIL import Image

    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    img = image.convert("RGB").resize((CFG.image_size, CFG.image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None, :]


def predict(image, session=None) -> dict:
    session = session or _session()
    x = _preprocess(image)
    logits = session.run(None, {"input": x})[0][0]
    e = np.exp(logits - logits.max())
    prob_ai = float((e / e.sum())[1])
    return {"ai_generated_prob": prob_ai}


if __name__ == "__main__":
    print({"best_val_auc": train()})
