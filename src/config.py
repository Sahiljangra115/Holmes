"""Single source of truth for paths, model names, thresholds, and the label mapping.

Every other module imports from here so there is exactly one place to change a
setting. Keep this file free of heavy imports so it stays cheap to load on a
Lambda cold start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root is two levels up from this file (src/config.py -> repo root).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MLRUNS_DIR = ROOT / "mlruns"


@dataclass(frozen=True)
class Config:
    # Reproducibility.
    seed: int = 42

    # Text model.
    text_model_name: str = "microsoft/deberta-v3-base"
    text_fallback_name: str = "roberta-base"
    text_max_len: int = 256
    text_epochs: int = 3
    text_lr: float = 2e-5
    text_batch_size: int = 16

    # LIAR2 6-way label ids: 0 pants-fire, 1 false, 2 barely-true,
    # 3 half-true, 4 mostly-true, 5 true.
    # Documented mapping to binary: the lower half of the truth scale is
    # "not_credible" (0), the upper half is "credible" (1). This collapse is a
    # judgement call and is stated in the README so the metric is honest.
    text_num_labels: int = 2
    liar2_to_binary: dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    )
    text_label_names: tuple[str, str] = ("not_credible", "credible")

    # Image model.
    image_model_name: str = "efficientnet_b0"
    image_size: int = 224
    image_epochs: int = 5
    image_lr: float = 1e-4
    image_batch_size: int = 32

    # Fusion weights. They sum to 1.0 over the AVAILABLE signals only, so an
    # unavailable signal (for example SynthID on the free tier) cannot move the
    # score. See fusion.py.
    w_text: float = 0.45
    w_image: float = 0.40
    w_meta: float = 0.10
    w_watermark: float = 0.05

    # Band thresholds on the fused authenticity confidence in [0, 1].
    band_synthetic_below: float = 0.35
    band_authentic_above: float = 0.65

    synthid_available: bool = False
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.1:8b"

    # Artifact paths.
    text_model_dir: Path = ARTIFACT_DIR / "text_model"
    image_onnx_path: Path = ARTIFACT_DIR / "image_detector.onnx"
    calibration_path: Path = ARTIFACT_DIR / "text_temperature.json"

    train_parquet: Path = DATA_DIR / "text_train.parquet"
    val_parquet: Path = DATA_DIR / "text_val.parquet"
    test_parquet: Path = DATA_DIR / "text_test.parquet"
    image_manifest: Path = DATA_DIR / "image_manifest.parquet"


CFG = Config()
