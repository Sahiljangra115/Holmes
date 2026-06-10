"""API contract with the heavy models monkeypatched out, so it runs offline."""
import types

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_text_contract(monkeypatch):
    fake = types.SimpleNamespace(label="credible", label_id=1, confidence=0.83)
    monkeypatch.setattr("src.models.text_classifier.predict", lambda text: fake)
    r = client.post("/predict/text", json={"text": "the sky is blue"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "credible"
    assert 0.0 <= body["confidence"] <= 1.0


def test_fuse_contract_text_only(monkeypatch):
    fake = types.SimpleNamespace(label="credible", label_id=1, confidence=0.83)
    monkeypatch.setattr("src.models.text_classifier.predict", lambda text: fake)
    # Avoid the network call in explain().
    monkeypatch.setattr("src.models.fusion.explain", lambda fused: "stub explanation")
    r = client.post("/predict/fuse", json={"text": "a claim", "image_base64": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["band"] in {"likely_authentic", "uncertain", "likely_synthetic"}
    assert 0.0 < body["confidence"] < 1.0
    assert "text" in body["used_signals"]
