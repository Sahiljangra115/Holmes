# Holmes

**AI-content provenance and misinformation detector.** Given text and/or an image, Holmes returns a calibrated confidence and a plain-language explanation of whether the content is likely AI-generated or misleading. It is deliberately not a binary fake/real oracle: it fuses four independent signals and reports how sure it actually is.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![DeBERTa-v3](https://img.shields.io/badge/text-DeBERTa--v3-yellow.svg)](https://huggingface.co/microsoft/deberta-v3-base)
[![ONNX](https://img.shields.io/badge/inference-ONNX-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![AWS Lambda](https://img.shields.io/badge/deploy-AWS%20Lambda-FF9900.svg)](https://aws.amazon.com/lambda/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

---

## The problem

"Is this fake?" is the wrong question, because no single detector answers it reliably. Image detectors break when the generator changes. Metadata is trivially stripped or forged. Text credibility is subjective. So instead of one overconfident classifier, Holmes runs four weak-but-honest signals, fuses them with renormalized weights, and reports a calibrated confidence with an uncertainty band. The output is an estimate, not a verdict.

## How it decides

Each signal returns an authenticity score in `[0, 1]` (0 = synthetic, 1 = authentic). Signals unavailable for a given request are dropped, and the remaining weights are renormalized, so a missing signal never silently counts as evidence.

| Signal | Weight | Method | Why it is weighted this way |
|--------|:------:|--------|-----------------------------|
| Text | 45% | `microsoft/deberta-v3-base` fine-tuned on a credibility corpus, temperature-calibrated | Strongest signal; confidence reflects real accuracy, not raw softmax |
| Image | 40% | `EfficientNet-B0` exported to ONNX | Reliable but generator-sensitive; reported with explicit false-positive rate |
| Metadata | 10% | Deterministic C2PA + EXIF parsing, no ML | Useful nudge, but metadata can be stripped or forged |
| Watermark | 5% | Google SynthID interface | Decisive when present, but unavailable on the free tier today |

## What sets it apart

- **Calibrated, not confident.** The text model minimizes Expected Calibration Error via temperature scaling. When it says 70%, it means it.
- **A baseline that must be beaten.** A TF-IDF and logistic-regression model is the anchor; the transformer ships only if it beats that baseline F1 on the same test set.
- **Graceful degradation.** Text-only request? The image and watermark signals drop out and weights renormalize. No signals at all? Confidence defaults to 0.5, not a fabricated answer.
- **No hard verdicts.** Confidence is clamped to `[0.01, 0.99]` and always paired with a band: `likely_authentic`, `uncertain`, or `likely_synthetic`.
- **LLM as narrator, never judge.** An optional local model narrates the numbers into prose; the code decides the band. If the model is unreachable, a deterministic template takes over.

## Architecture

```
                 Request (text and/or image)
                            |
          +-----------------+-----------------+
          v                 v                 v
   +------------+    +------------+    +------------+    +------------+
   |   Text     |    |   Image    |    |  Metadata  |    |  Watermark |
   | DeBERTa-v3 |    | EffNet-B0  |    | C2PA+EXIF  |    |  SynthID   |
   | calibrated |    |  (ONNX)    |    | determ.    |    | (unavail.) |
   +-----+------+    +-----+------+    +-----+------+    +-----+------+
         +-----------------+----- weighted fusion ------------+
                           |  (drop unavailable, renormalize)
                           v
                  confidence in [0.01, 0.99]  ->  band
                           |
                           v
                LLM narration (optional, grounded)
                           |
                           v
                     JSON response
```

## API

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/health` | none | status |
| POST | `/predict/text` | `{ "text": "..." }` | label + confidence |
| POST | `/predict/image` | `{ "image_base64": "..." }` | AI-generated probability + provenance |
| POST | `/predict/fuse` | `{ "text": "...", "image_base64": "..." }` | confidence, band, explanation, used signals, per-signal breakdown |

Example fused response:

```json
{
  "confidence": 0.31,
  "band": "likely_synthetic",
  "explanation": "The text model leans uncredible and the image model sees AI artifacts...",
  "used_signals": ["text", "image", "metadata"],
  "breakdown": { "text": 0.28, "image": 0.30, "metadata": 0.50 },
  "disclaimer": "This is an estimate, not an oracle."
}
```

## Quickstart

```bash
python scripts/check_env.py            # environment / GPU check

# Data
python -m src.data.prepare_text        # downloads + maps the credibility corpus
python -m src.data.prepare_images /path/to/image-dataset

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

## Evaluation

Honest metrics, not headline numbers. Run the eval scripts on your trained artifacts to fill the table. False-positive rate is reported explicitly, because flagging a real photo as AI has a real cost.

| Metric | Script | Value |
|--------|--------|-------|
| Text F1 (macro) | `eval_text.py` | run to fill |
| Text ECE (calibration error) | `eval_text.py` | run to fill |
| Image ROC-AUC | `eval_image.py` | run to fill |
| Image false-positive rate | `eval_image.py` | run to fill |
| Fusion band agreement | `eval_fusion.py` | run to fill |

## Deployment

Container-first, built for the AWS Lambda free tier (Mangum adapts FastAPI to the Lambda runtime). Large artifacts live in S3 and are pulled to `/tmp` on cold start; models lazy-load only when their endpoint is hit. A least-privilege IAM policy grants read-only S3 on the artifact prefix plus CloudWatch logs. Full walkthrough in `infra/deploy_lambda.md`. Trade-off: free-tier cold starts add a few seconds on the first request.

## Project layout

```
.
├── src/
│   ├── config.py            # single source of truth: paths, weights, thresholds
│   ├── models/              # text_classifier, baseline_text, image_detector,
│   │                        # metadata_provenance, synthid_client, fusion
│   ├── api/                 # main (FastAPI), schemas (Pydantic), handler (Lambda)
│   ├── data/                # text + image dataset preparation
│   └── eval/                # eval_text, eval_image, eval_fusion, results_table
├── tests/                   # api, calibration, fusion, metadata, data
├── ui/index.html            # dashboard: confidence dial + per-signal bars
├── infra/                   # deploy_lambda.md, iam-policy.json
├── .github/workflows/ci.yml # lint (ruff) + test (pytest) + docker build
├── Dockerfile               # Lambda-compatible image
└── pyproject.toml
```

## Limitations

- Image detection degrades when the generator distribution shifts; the explicit FPR is there to keep that visible.
- The credibility labels are annotations, not ground truth of falsehood.
- SynthID is unavailable on the free tier, so the watermark signal is wired but inert.
- Metadata is forgeable, which is exactly why it carries only 10% of the weight.
- Lambda free tier means cold-start latency on the first request.

## License

MIT.
