"""Write the README results table from a metrics dict. Keeps the README honest:
the numbers come from eval scripts, not from hand-editing.

    python -m src.eval.results_table
"""
from __future__ import annotations

from src.config import CFG

MARK_START = "<!-- RESULTS:START -->"
MARK_END = "<!-- RESULTS:END -->"


def render_table(metrics: dict) -> str:
    rows = ["| metric | value |", "| --- | --- |"]
    for k, v in metrics.items():
        rows.append(f"| {k} | {v} |")
    return "\n".join(rows)


def write_results(metrics: dict, readme=None) -> None:
    readme = readme or (CFG.train_parquet.parent.parent / "README.md")
    table = f"{MARK_START}\n{render_table(metrics)}\n{MARK_END}"
    text = readme.read_text() if readme.exists() else "# Y1\n"
    if MARK_START in text and MARK_END in text:
        pre = text.split(MARK_START)[0]
        post = text.split(MARK_END)[1]
        text = pre + table + post
    else:
        text += "\n\n## Results\n" + table + "\n"
    readme.write_text(text)


if __name__ == "__main__":
    write_results({"text_f1_macro": "fill me", "image_auc": "fill me", "image_fpr": "fill me"})
