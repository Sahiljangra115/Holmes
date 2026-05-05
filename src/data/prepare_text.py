"""Download LIAR2, map the 6-way truth scale to a binary credibility target,
make a seeded train/val/test split, and save parquet files.

LIAR2 already ships train/validation/test splits, so we keep them (no re-split)
and only clean and re-label. The split is therefore deterministic by
construction. Run once:

    python -m src.data.prepare_text
"""
from __future__ import annotations

import pandas as pd

from src.config import CFG

# The LIAR2 statement text lives in the "statement" column and the 6-way label
# in "label". Other columns (speaker, context, justification) are kept for
# optional richer experiments but are not required by the baseline.
TEXT_COL = "statement"
LABEL_COL = "label"


def _to_frame(split) -> pd.DataFrame:
    df = split.to_pandas()
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.strip().astype(bool)]
    # Collapse 6-way to binary using the documented mapping in config.
    df["label"] = df[LABEL_COL].map(CFG.liar2_to_binary).astype("int64")
    df["text"] = df[TEXT_COL].str.strip()
    return df[["text", "label"]].reset_index(drop=True)


def prepare() -> dict[str, int]:
    import datasets

    ds = datasets.load_dataset("chengxuphd/liar2")
    CFG.train_parquet.parent.mkdir(parents=True, exist_ok=True)

    # HF split is named "validation", not "val".
    splits = {
        CFG.train_parquet: ds["train"],
        CFG.val_parquet: ds["validation"],
        CFG.test_parquet: ds["test"],
    }
    counts: dict[str, int] = {}
    for path, split in splits.items():
        frame = _to_frame(split)
        frame.to_parquet(path, index=False)
        counts[path.name] = len(frame)
    return counts


if __name__ == "__main__":
    print(prepare())
