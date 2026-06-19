"""Provenance reader: C2PA manifest + EXIF. Deterministic parsing, no ML.

Absence of provenance is NOT evidence of forgery, it is just absence. The caller
must treat a missing manifest as "unknown", which fusion does.
"""
from __future__ import annotations

from typing import Any


def _read_c2pa(image_path: str) -> dict[str, Any]:
    try:
        import c2pa

        reader = c2pa.Reader.from_file(image_path)
        manifest = reader.json()
        return {"has_c2pa": True, "manifest": manifest}
    except Exception:
        return {"has_c2pa": False, "manifest": None}


def _read_exif(image_path: str) -> dict[str, Any]:
    try:
        from PIL import Image, ExifTags

        img = Image.open(image_path)
        raw = img.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        return {
            "camera_make": tags.get("Make"),
            "camera_model": tags.get("Model"),
            "software": tags.get("Software"),
        }
    except Exception:
        return {"camera_make": None, "camera_model": None, "software": None}


def extract_metadata(image_path: str) -> dict[str, Any]:
    c2pa_info = _read_c2pa(image_path)
    exif = _read_exif(image_path)

    inconsistencies: list[str] = []
    sw = (exif.get("software") or "").lower()
    if sw and not exif.get("camera_make"):
        inconsistencies.append("editing software present without camera make")

    return {
        "has_c2pa": c2pa_info["has_c2pa"],
        "signer": None,
        "camera_make": exif.get("camera_make"),
        "software": exif.get("software"),
        "inconsistencies": inconsistencies,
    }
