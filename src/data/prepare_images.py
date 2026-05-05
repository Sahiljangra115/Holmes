"""Build a seeded train/val/test manifest for a real-vs-AI image set.

Expected on-disk layout (CIFAKE style), one folder per class:

    <root>/REAL/*.jpg
    <root>/FAKE/*.jpg

We do NOT copy images. We build a manifest (path, label, split) so the training
loop can read straight from disk. The split is seeded so it is reproducible and
never re-rolled. Keep generators diverse inside FAKE (SD, Midjourney style,
DALLE style) so the detector does not learn one generator fingerprint.

    python -m src.data.prepare_images /path/to/cifake
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG

LABELS = {"REAL": 0, "FAKE": 1}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_manifest(
    root: str | Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    out: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    rows: list[dict] = []
    for name, label in LABELS.items():
        folder = root / name
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in EXTS:
                rows.append({"path": str(p), "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no images found under {root} (expected REAL/ and FAKE/)")

    rng = np.random.default_rng(CFG.seed)
    df = df.sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)
    u = rng.random(len(df))
    split = np.where(u < test_frac, "test", np.where(u < test_frac + val_frac, "val", "train"))
    df["split"] = split

    out = Path(out) if out else CFG.image_manifest
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else str(CFG.train_parquet.parent / "cifake")
    out = build_manifest(root)
    print(out["split"].value_counts().to_dict())
