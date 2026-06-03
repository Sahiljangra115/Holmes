"""SynthID watermark check, integration interface only.

On the free tier there is no API access, so this returns UNAVAILABLE. The shape
is fixed so a real client can drop in later without changing fusion. Fusion must
treat UNAVAILABLE as "no information", never as evidence either way.
"""
from __future__ import annotations

from typing import Any

from src.config import CFG


def check_watermark(image: Any) -> dict[str, Any]:
    if not CFG.synthid_available:
        return {"status": "UNAVAILABLE", "detail": "SynthID API not configured on this deployment"}
    # ponytail: real client would POST the image bytes here, no creds on free tier
    raise NotImplementedError("enable a real SynthID client and set synthid_available=True")
