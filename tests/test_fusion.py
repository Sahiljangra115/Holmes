"""Fusion logic: bands, and the rule that an unavailable signal cannot swing
the score. These are the load-bearing guarantees, so they get real tests."""
from src.models import fusion


def _avail(score, source):
    return {"score": score, "available": True, "source": source}


def _na(source):
    return {"score": None, "available": False, "source": source}


def test_unavailable_watermark_does_not_swing_score():
    t = _avail(0.9, "text")
    img = _avail(0.9, "image")
    meta = _na("metadata")
    wm_absent = _na("watermark")
    wm_present_na = fusion.watermark_signal({"status": "UNAVAILABLE"})

    a = fusion.fuse(t, img, meta, wm_absent)
    b = fusion.fuse(t, img, meta, wm_present_na)
    assert a["confidence"] == b["confidence"]
    assert "watermark" not in a["used_signals"]


def test_high_authenticity_gives_authentic_band():
    out = fusion.fuse(_avail(0.95, "text"), _avail(0.9, "image"), _na("metadata"), _na("watermark"))
    assert out["band"] == "likely_authentic"


def test_low_authenticity_gives_synthetic_band():
    out = fusion.fuse(_avail(0.05, "text"), _avail(0.1, "image"), _na("metadata"), _na("watermark"))
    assert out["band"] == "likely_synthetic"


def test_never_hard_zero_or_one():
    out = fusion.fuse(_avail(1.0, "text"), _avail(1.0, "image"), _na("metadata"), _na("watermark"))
    assert 0.0 < out["confidence"] < 1.0


def test_no_signals_is_uncertain():
    out = fusion.fuse(_na("text"), _na("image"), _na("metadata"), _na("watermark"))
    assert out["band"] == "uncertain"
    assert out["confidence"] == 0.5
