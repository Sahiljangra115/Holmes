"""FastAPI service. Models are loaded lazily and cached so a Lambda cold start
only pays for what an endpoint actually needs.

Run locally:  uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import base64
import io

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    FuseRequest,
    FuseResponse,
    ImageRequest,
    ImageResponse,
    TextRequest,
    TextResponse,
)

app = FastAPI(title="Y1 Provenance & Misinformation Detector")


def _decode_image(image_base64: str):
    from PIL import Image

    try:
        raw = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad image: {exc}") from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/text", response_model=TextResponse)
def predict_text(req: TextRequest):
    from src.models.text_classifier import predict

    p = predict(req.text)
    return TextResponse(label=p.label, confidence=p.confidence)


@app.post("/predict/image", response_model=ImageResponse)
def predict_image(req: ImageRequest):
    from src.models.image_detector import predict
    from src.models.metadata_provenance import extract_metadata

    img = _decode_image(req.image_base64)
    out = predict(img)
    return ImageResponse(ai_generated_prob=out["ai_generated_prob"], provenance={})


@app.post("/predict/fuse", response_model=FuseResponse)
def predict_fuse(req: FuseRequest):
    from src.models import fusion
    from src.models.synthid_client import check_watermark

    text_pred = None
    if req.text.strip():
        from src.models.text_classifier import predict as text_predict

        text_pred = text_predict(req.text)

    image_out = None
    if req.image_base64.strip():
        from src.models.image_detector import predict as image_predict

        image_out = image_predict(_decode_image(req.image_base64))

    t = fusion.text_signal(text_pred)
    img = fusion.image_signal(image_out)
    meta = fusion.meta_signal(None)
    wm = fusion.watermark_signal(check_watermark(req.image_base64 or None))

    fused = fusion.fuse(t, img, meta, wm)
    return FuseResponse(explanation=fusion.explain(fused), **fused)
