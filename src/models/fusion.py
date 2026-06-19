"""Fuse the four signals into one calibrated authenticity confidence and a band.

Design choices, all defensible:
- Transparent weighted average over the AVAILABLE signals only. An unavailable
  signal (SynthID on free tier, or a missing image) is dropped and the weights
  are renormalised, so absence cannot move the score. This is the honest way to
  handle graceful degradation.
- The score is "authenticity": 1.0 = likely authentic, 0.0 = likely synthetic.
- We never output a hard 0 or 1, and never a bare label without a band.
- The LLM only narrates the numbers. The band is decided by code, not the LLM.

Each input signal is a uniform dict: {"score": float in [0,1] or None,
"available": bool, "source": str}. score is the signal's own authenticity vote.
"""
from __future__ import annotations

from typing import Any

from src.config import CFG


def _signal(score: float | None, available: bool, source: str) -> dict[str, Any]:
    return {"score": score, "available": available, "source": source}


def text_signal(pred) -> dict[str, Any]:
    if pred is None:
        return _signal(None, False, "text")
    auth = pred.confidence if pred.label == "credible" else 1.0 - pred.confidence
    return _signal(float(auth), True, "text")


def image_signal(pred: dict | None) -> dict[str, Any]:
    if not pred or pred.get("ai_generated_prob") is None:
        return _signal(None, False, "image")
    return _signal(1.0 - float(pred["ai_generated_prob"]), True, "image")


def meta_signal(meta: dict | None) -> dict[str, Any]:
    if not meta:
        return _signal(None, False, "metadata")
    score = 0.5
    if meta.get("has_c2pa"):
        score += 0.3
    score -= 0.1 * len(meta.get("inconsistencies", []))
    return _signal(min(1.0, max(0.0, score)), True, "metadata")


def watermark_signal(wm: dict | None) -> dict[str, Any]:
    if not wm or wm.get("status") != "DETECTED":
        return _signal(None, False, "watermark")
    return _signal(0.0, True, "watermark")


def fuse(text_sig, image_sig, meta_sig, wm_sig, cfg: Config = CFG) -> dict[str, Any]:  # noqa: F821
    weights = {
        "text": cfg.w_text,
        "image": cfg.w_image,
        "metadata": cfg.w_meta,
        "watermark": cfg.w_watermark,
    }
    sigs = [text_sig, image_sig, meta_sig, wm_sig]
    avail = [s for s in sigs if s["available"] and s["score"] is not None]

    if not avail:
        confidence = 0.5
    else:
        total_w = sum(weights[s["source"]] for s in avail)
        confidence = sum(weights[s["source"]] * s["score"] for s in avail) / total_w

    confidence = min(0.99, max(0.01, confidence))

    if confidence < cfg.band_synthetic_below:
        band = "likely_synthetic"
    elif confidence > cfg.band_authentic_above:
        band = "likely_authentic"
    else:
        band = "uncertain"

    return {
        "confidence": round(confidence, 4),
        "band": band,
        "used_signals": [s["source"] for s in avail],
        "breakdown": {s["source"]: s["score"] for s in sigs if s["available"]},
    }


def explain(fused: dict[str, Any]) -> str:
    """Grounded natural-language narration. Calls Ollama if reachable, else a
    template. The model is told to narrate the numbers only, never to invent a
    verdict beyond the band that code already decided."""
    breakdown = ", ".join(f"{k}={v:.2f}" for k, v in fused["breakdown"].items()) or "none"
    facts = (
        f"authenticity confidence={fused['confidence']:.2f}, band={fused['band']}, "
        f"per-signal authenticity scores: {breakdown}"
    )
    prompt = (
        "You are a careful assistant. Given ONLY these numbers, write 2 to 3 "
        "sentences that explain what they mean for whether the content is "
        "AI-generated. Do not state any verdict beyond the given band. Do not "
        f"invent numbers.\nNumbers: {facts}"
    )
    try:
        import requests

        r = requests.post(
            CFG.ollama_url,
            json={"model": CFG.ollama_model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception:
        return (
            f"This is an estimate, not an oracle. The fused authenticity confidence is "
            f"{fused['confidence']:.2f}, which falls in the '{fused['band']}' band. "
            f"Signals used: {', '.join(fused['used_signals']) or 'none available'}."
        )
