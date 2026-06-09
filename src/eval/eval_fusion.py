"""End-to-end fusion eval on a small hand-built set with known ground truth.

The eval set is a JSON list of items, each: {"text": str|null, "image": path|null,
"label": "authentic"|"synthetic"}. Reports band agreement with the label.

    python -m src.eval.eval_fusion eval_set.json
"""
from __future__ import annotations

import json
import sys


def evaluate(eval_set_path: str) -> dict:
    from PIL import Image

    from src.models import fusion
    from src.models.image_detector import predict as image_predict
    from src.models.metadata_provenance import extract_metadata
    from src.models.synthid_client import check_watermark
    from src.models.text_classifier import predict as text_predict

    items = json.loads(open(eval_set_path).read())
    correct = 0
    for it in items:
        t = fusion.text_signal(text_predict(it["text"]) if it.get("text") else None)
        img = fusion.image_signal(image_predict(Image.open(it["image"])) if it.get("image") else None)
        meta = fusion.meta_signal(extract_metadata(it["image"]) if it.get("image") else None)
        wm = fusion.watermark_signal(check_watermark(it.get("image")))
        fused = fusion.fuse(t, img, meta, wm)
        want_auth = it["label"] == "authentic"
        got_auth = fused["band"] == "likely_authentic"
        got_synth = fused["band"] == "likely_synthetic"
        if (want_auth and got_auth) or (not want_auth and got_synth):
            correct += 1
    return {"n": len(items), "band_agreement": correct / max(1, len(items))}


if __name__ == "__main__":
    print(evaluate(sys.argv[1]))
