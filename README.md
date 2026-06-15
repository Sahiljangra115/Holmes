# Y1: AI-Content Provenance & Misinformation Detector

> Given text and/or an image, the service returns a **calibrated** confidence and a plain-language explanation of whether the content is likely AI-generated or misleading. It is deliberately not a binary fake/real oracle. It fuses four independent signals and tells you honestly how sure it is.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/DeBERTa--v3-yellow.svg)](https://huggingface.co/microsoft/deberta-v3-base)
[![ONNX](https://img.shields.io/badge/ONNX-runtime-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![AWS Lambda](https://img.shields.io/badge/deploy-AWS%20Lambda-FF9900.svg)](https://aws.amazon.com/lambda/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

---

## The idea

"Is this fake?" is the wrong question, because no single detector is reliable enough to answer it. AI-image detectors break when the generator changes. Metadata can be stripped or forged. Text credibility is subjective. So instead of one overconfident classifier, this system runs **four weak-but-honest signals**, fuses them with renormalized weights, and reports a calibrated confidence with an uncertainty band. The output is an estimate, not a verdict.

## The four signals

Each signal returns an authenticity score in `[0, 1]` (0 = synthetic, 1 = authentic). Signals that are unavailable for a given request are dropped, and the remaining weights are renormalized so a missing signal never silently counts as evidence.

| Signal | Weight | Model / method | Honesty note |
|--------|:------:|----------------|--------------|
| **Text** | 45% | `microsoft/deberta-v3-base` fine-tuned on LIAR2, temperature-calibrated | Confidence reflects actual accuracy, not raw softmax overconfidence |
| **Image** | 40% | `EfficientNet-B0` exported to ONNX | Reported with explicit false-positive rate; selected by validation AUC, not accuracy |
| **Metadata** | 10% | Deterministic C2PA + EXIF parsing, no ML | Deliberately weak; metadata can be stripped or forged |
| **Watermark** | 5% | Google SynthID interface | Currently unavailable on the free tier; wired but off |

## Why this is more than a classifier

- **Calibration first.** The text model minimizes Expected Calibration Error via temperature scaling. When it says 70% it means it, instead of the inflated certainty raw softmax gives you.
- **A baseline that must be beaten.** A TF-IDF + logistic regression model is trained as an anchor. The DeBERTa transformer ships only if it beats that baseline F1 on the same test set. No "the big model is obviously better" hand-waving.
- **Graceful degradation.** Text-only request? The image and watermark signals drop out and weights renormalize. No signals at all? Confidence defaults to 0.5 (uncertain), not a fabricated answer.
- **No hard verdicts.** Confidence is clamped to `[0.01, 0.99]` and always paired with a band: `likely_authentic`, `uncertain`, or `likely_synthetic`.
- **LLM as narrator, never as judge.** An optional local Ollama model narrates the numbers into prose. The code decides the band; the LLM is forbidden from inventing a verdict. If Ollama is unreachable, a deterministic template takes over.

## Architecture

```
                 Request (text and/or image)
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │   Text     │    │   Image    │    │  Metadata  │    │  Watermark │
   │ DeBERTa-v3 │    │ EffNet-B0  │    │ C2PA+EXIF  │    │  SynthID   │
   │ calibrated │    │  (ONNX)    │    │determinist.│    │ (unavail.) │
   └─────┬──────┘    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
         └─────────────────┴───── weighted fusion ────────────┘
                            │  (drop unavailable, renormalize)
                            ▼
                   confidence ∈ [0.01, 0.99]  →  band
                            │
                            ▼
                 LLM narration (Ollama, optional)
                            │
                            ▼
                      JSON response
```

## API

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict/text` | `{"text": "..."}` | label + confidence |
| `POST` | `/predict/image` | `{"image_base64": "..."}` | AI-generated probability + provenance |
| `POST` | `/predict/fuse` | `{"text": "...", "image_base64": "..."}` | confidence, band, explanation, used signals, per-signal breakdown, disclaimer |

Example fused response:

```json
{
  "confidence": 0.31,
  "band": "likely_synthetic",
  "explanation": "The text model leans uncredible and the image model sees AI artifacts...",
  "used_signals": ["text", "image", "metadata"],
  "breakdown": {"text": 0.28, "image": 0.30, "metadata": 0.50},
  "disclaimer": "This is an estimate, not an oracle."
}
```

## Quickstart

```bash
python scripts/check_env.py            # GPU / environment check

# Data
python -m src.data.prepare_text                  # downloads LIAR2 (~13k claims)
python -m src.data.prepare_images /path/to/cifake

# Train
python -m src.models.baseline_text     # TF-IDF anchor
python -m src.models.text_classifier   # fine-tune DeBERTa + calibrate (GPU)
python -m src.models.image_detector    # fine-tune EfficientNet, export ONNX (GPU)

# Evaluate
python -m src.eval.eval_text           # F1, ECE, confusion matrix
python -m src.eval.eval_image          # accuracy, ROC-AUC, false-positive rate
python -m src.eval.eval_fusion eval_set.json

# Serve
uvicorn src.api.main:app --reload      # http://localhost:8000/docs

pytest                                 # offline, CPU-friendly, models mocked
```

## Datasets

- **LIAR2** (text): ~13k news claims with six credibility labels, mapped to binary `credible` / `not_credible`. Downloaded via Hugging Face `datasets`. Seed 42 for reproducible splits.
- **CIFAKE** (images): real-vs-AI images, supplied as a local `REAL/` `FAKE/` directory; the prep script builds a parquet manifest with deterministic train/val/test splits.

## Evaluation metrics

The repo reports honest metrics, not headline numbers. Run the eval scripts to populate the table for your trained artifacts:

| Metric | Where | Value |
|--------|-------|-------|
| Text F1 (macro) | `eval_text.py` | run to fill |
| Text ECE (calibration error) | `eval_text.py` | run to fill |
| Image ROC-AUC | `eval_image.py` | run to fill |
| Image false-positive rate | `eval_image.py` | run to fill |
| Fusion band agreement | `eval_fusion.py` | run to fill |

False-positive rate is reported explicitly because flagging a real photo as AI has a real cost.

## Deployment

Container-first, designed for the AWS Lambda free tier (Mangum adapts FastAPI to the Lambda runtime). Large model artifacts live in S3 and are pulled to `/tmp` on cold start; models lazy-load only when their endpoint is hit. A least-privilege IAM policy (`infra/iam-policy.json`) grants read-only S3 on the artifact prefix plus CloudWatch logs. Full walkthrough in `infra/deploy_lambda.md`.

Trade-off: free-tier cold starts add a few seconds on the first request. Upgrade path is provisioned concurrency or a small always-on instance.

## Project layout

```
y1-provenance/
├── src/
│   ├── config.py            # single source of truth: paths, weights, thresholds
│   ├── models/              # text_classifier, baseline_text, image_detector,
│   │                        # metadata_provenance, synthid_client, fusion
│   ├── api/                 # main (FastAPI), schemas (Pydantic), handler (Lambda)
│   ├── data/                # prepare_text (LIAR2), prepare_images (CIFAKE)
│   └── eval/                # eval_text, eval_image, eval_fusion, results_table
├── tests/                   # api, calibration, fusion, metadata, data
├── ui/index.html            # dashboard: SVG confidence dial + per-signal bars
├── infra/                   # deploy_lambda.md, iam-policy.json
├── .github/workflows/ci.yml # lint (ruff) + test (pytest) + docker build
├── Dockerfile               # Lambda-compatible image
└── pyproject.toml
```

## Honest limitations

- AI-image detection degrades when the generator distribution shifts; the explicit FPR is there to remind you of that.
- LIAR2 labels are credibility annotations, not ground truth of falsehood.
- SynthID is unavailable on the free tier, so the watermark signal is wired but inert.
- Metadata is forgeable, which is exactly why it carries only 10% weight.
- Lambda free tier means cold-start latency.

---

The thesis of this project: a detector that admits uncertainty and renormalizes around missing evidence is more useful than a confident classifier that is wrong 30% of the time and never says so.
