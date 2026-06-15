# Y1 Code Walkthrough Questions

Per-line, grounded in your actual `src/` code. The "explain this line" questions
that prove you understand your own provenance pipeline, not just that it runs.

Recurring character: **Priya**, a newsroom fact-checker who pastes a suspect
article and uploads a possibly-AI image to decide whether to publish.

Format: answer in your own words, leave Verdict/Explanation for grading.

---

## Section A: config and the fusion weights

### A1: Weights that sum over available signals
* Question: `config.py` sets `w_text=0.45, w_image=0.40, w_meta=0.10, w_watermark=0.05`. The fusion docstring says weights are renormalized "over the AVAILABLE signals only". Why renormalize instead of treating a missing signal as a 0 score? What would a literal 0 do to Priya's authenticity confidence when SynthID is simply unavailable?
* Answer:
* Verdict:
* Explanation:

### A2: The two band thresholds
* Question: `band_synthetic_below=0.35` and `band_authentic_above=0.65` leave a deliberate gap. What is in that gap, and why is having an explicit "uncertain" band more honest than a single 0.5 cutoff that forces every input into authentic-or-synthetic?
* Answer:
* Verdict:
* Explanation:

### A3: synthid_available defaults False
* Question: `CFG.synthid_available = False`. Trace what `synthid_client.check_watermark` returns because of this flag, and how that propagates through `watermark_signal` into `fuse`. Why is "the watermark contributes nothing" the correct behavior on the free tier rather than "no watermark means authentic"?
* Answer:
* Verdict:
* Explanation:

---

## Section B: the fusion math (the heart of the project)

### B1: text_signal maps onto an authenticity axis
* Question: `text_signal` does `auth = pred.confidence if pred.label == "credible" else 1.0 - pred.confidence`. Walk a concrete case: the classifier says "not_credible" with confidence 0.8. What authenticity score does that become, and why the `1.0 - confidence` flip?
* Answer:
* Verdict:
* Explanation:

### B2: meta_signal is a weak nudge
* Question: `meta_signal` starts at 0.5, adds 0.3 for C2PA, subtracts 0.1 per inconsistency, then clamps to [0,1]. Why does metadata start at a neutral 0.5 and only nudge, rather than acting as a strong vote? What property of metadata (it can be stripped or forged) justifies the small weight?
* Answer:
* Verdict:
* Explanation:

### B3: watermark_signal returns 0.0 when detected
* Question: A detected SynthID watermark makes `watermark_signal` return score 0.0 (fully synthetic). Why 0.0 and not, say, 0.1? What is special about a detected generative watermark versus the other three soft signals?
* Answer:
* Verdict:
* Explanation:

### B4: the renormalization line
* Question: In `fuse`, `total_w = sum(weights[s["source"]] for s in avail)` then `confidence = sum(weights[...]*s["score"] for s in avail) / total_w`. If only text (0.45) and image (0.40) are available, what is the effective weight each gets after dividing by total_w? Show the arithmetic.
* Answer:
* Verdict:
* Explanation:

### B5: the clamp away from 0 and 1
* Question: `confidence = min(0.99, max(0.01, confidence))`. Why refuse to ever output a hard 0.0 or 1.0 authenticity? What claim are you avoiding making, and how does this tie to the project's "estimate, not an oracle" framing?
* Answer:
* Verdict:
* Explanation:

### B6: the empty-avail fallback
* Question: `if not avail: confidence = 0.5`. When does `avail` end up empty, and why is 0.5 (maximum uncertainty) the only defensible number when zero signals are available? What would returning 0.0 or 1.0 here imply?
* Answer:
* Verdict:
* Explanation:

---

## Section C: the LLM only narrates

### C1: code decides the band, not the model
* Question: `fuse` computes `band` from thresholds; `explain` then narrates it. The `explain` prompt says "Do not state any verdict beyond the given band." Why is it a design rule that the LLM never decides the verdict? What failure are you preventing by making the LLM a narrator, not a judge?
* Answer:
* Verdict:
* Explanation:

### C2: the try/except fallback in explain
* Question: `explain` POSTs to Ollama inside a `try`, and on any exception returns a template string. Why fall back to a deterministic template instead of letting the request fail? What does Priya get when Ollama is down, and is it still truthful?
* Answer:
* Verdict:
* Explanation:

### C3: the breakdown string is built from numbers only
* Question: `breakdown = ", ".join(f"{k}={v:.2f}" ...)` then the prompt says "Do not invent numbers." Why feed the model pre-formatted numbers and explicitly forbid invention, rather than handing it raw signal dicts and trusting it to format?
* Answer:
* Verdict:
* Explanation:

---

## Section D: text classifier and calibration

### D1: temperature scaling does not change argmax
* Question: `predict` does `_softmax(logits / temp)`. The comment says temperature scaling "does not change the argmax (so accuracy is unchanged) but makes the confidence honest." Explain why dividing all logits by the same T cannot change which class wins, yet does change the confidence number.
* Answer:
* Verdict:
* Explanation:

### D2: fitting T on validation, not train
* Question: `fit_temperature` is called on `val_logits`/`val_labels`, never on the training set. Why must temperature be fit on held-out data? What goes wrong if you calibrate on the same data the model trained on?
* Answer:
* Verdict:
* Explanation:

### D3: the grid search over T
* Question: `fit_temperature` builds `grid = np.linspace(0.25, 5.0, 96)` and picks the T with the lowest NLL. Why a grid search instead of gradient descent for a single scalar? What do the bounds 0.25 and 5.0 represent (under- vs over-confidence)?
* Answer:
* Verdict:
* Explanation:

### D4: load_temperature defaults to 1.0
* Question: If `calibration_path` does not exist, `load_temperature` returns 1.0. Why is 1.0 the correct default, and what does dividing logits by 1.0 do? When would Priya's deployment hit this default?
* Answer:
* Verdict:
* Explanation:

### D5: the LIAR2 to binary collapse
* Question: `config.py` maps 6 LIAR2 truth labels to binary via `{0:0,1:0,2:0,3:1,4:1,5:1}`. Which classes become "not_credible" and which become "credible"? Why is the README required to state this collapse for the accuracy number to be honest?
* Answer:
* Verdict:
* Explanation:

---

## Section E: metadata provenance (deterministic, no ML)

### E1: absence is not evidence
* Question: `_read_c2pa` returns `{"has_c2pa": False}` on any exception (missing library, unsigned asset). The module docstring insists "Absence of provenance is NOT evidence of forgery." Why is this distinction the whole point, and what bad product behavior follows if you treat "no manifest" as "fake"?
* Answer:
* Verdict:
* Explanation:

### E2: the inconsistency heuristic
* Question: `extract_metadata` flags "editing software present without camera make". Why is that only a weak smell and not a verdict? Give a legitimate photo that would trip this flag, and explain why it feeds a small nudge in `meta_signal` rather than a conclusion.
* Answer:
* Verdict:
* Explanation:

### E3: catch-all except returns neutral
* Question: Both `_read_c2pa` and `_read_exif` wrap everything in `try/except Exception` and return a neutral dict. What is the risk of such a broad catch (swallowing a real bug), and why is "never crash the provenance read" the priority that justifies it here?
* Answer:
* Verdict:
* Explanation:

---

## Section F: the API and serving

### F1: lazy model imports per endpoint
* Question: `predict_text` imports the model inside the function, not at module top. Connect this to a Lambda cold start: why does `/health` need to stay instant while `/predict/text` is allowed to pay the model load? What is the per-request cost you accept?
* Answer:
* Verdict:
* Explanation:

### F2: the fuse endpoint assembles all four signals
* Question: `predict_fuse` builds `t, img, meta, wm` then calls `fusion.fuse`. Notice `meta = fusion.meta_signal(None)` is passed None today. What does that mean for the metadata contribution on the base64-only path, and is it honest given the renormalization in B4?
* Answer:
* Verdict:
* Explanation:

### F3: _decode_image raises 400
* Question: `_decode_image` wraps decoding in try/except and raises `HTTPException(status_code=400, ...)`. Why 400 (client error) rather than 500 (server error) when Priya uploads a corrupt image? What is each status code telling the caller?
* Answer:
* Verdict:
* Explanation:

### F4: Mangum in handler.py
* Question: `handler.py` is `handler = Mangum(app)`. What does Mangum translate, and why is it the bridge that lets the same FastAPI app run under `uvicorn` locally and Lambda in production unchanged?
* Answer:
* Verdict:
* Explanation:

---

# Reliability and Scalability (appended)

Past "does it run" into "does it survive production". Same format.

## Section G: Reliability

### G1: Ollama as a flaky dependency
* Question: `explain` depends on a local Ollama at `CFG.ollama_url`. In production that is a network hop. The `timeout=20` bounds one call, but under load a slow Ollama serializes requests. What is your degradation story (the template fallback helps), and would you make narration async or optional?
* Answer:
* Verdict:
* Explanation:

### G2: model artifact missing at runtime
* Question: `text_classifier._load` reads `CFG.text_model_dir`. On a fresh deploy with no fine-tuned model, the load fails. What exception surfaces and where? How should `/predict/text` report "model not deployed" instead of a raw 500?
* Answer:
* Verdict:
* Explanation:

### G3: calibration drift
* Question: Temperature was fit once on a 2024 validation set. The distribution of articles Priya sees shifts (new topics, new generators). Why might your calibrated confidence quietly become dishonest over time, and how would you detect and refit T without retraining the whole model?
* Answer:
* Verdict:
* Explanation:

### G4: the fusion is only as honest as its weakest signal
* Question: If the image detector is poorly calibrated and outputs overconfident `ai_generated_prob`, the fused confidence inherits that. Since fusion is a weighted average, how does one miscalibrated signal poison the band? What would you monitor per-signal to catch it?
* Answer:
* Verdict:
* Explanation:

### G5: adversarial inputs
* Question: A bad actor strips C2PA and fakes EXIF to look like a real camera. Your metadata signal nudges toward authentic. Why is metadata structurally weak against adversaries, and why does the small `w_meta=0.10` weight limit the damage by design?
* Answer:
* Verdict:
* Explanation:

## Section H: Scalability

### H1: serving DeBERTa under load
* Question: The text model loads lazily and stays cached per container. Under bursty newsroom traffic, every cold Lambda reloads DeBERTa. When does this stop scaling, and what is the upgrade path (ONNX export like the sibling y2 food model, provisioned concurrency, a long-lived GPU container)?
* Answer:
* Verdict:
* Explanation:

### H2: batching predictions
* Question: Priya's newsroom submits 200 articles at once. Today each is one request, one forward pass. How would request batching cut cost and latency, and what changes in `predict` to accept a batch while keeping the single-item API?
* Answer:
* Verdict:
* Explanation:

### H3: the four signals scale differently
* Question: Text and image are model forward passes; metadata is cheap parsing; watermark is an external API. Under 1M requests/month, which signal dominates cost and which dominates latency? How would you cache or short-circuit the cheap signals to protect the expensive ones?
* Answer:
* Verdict:
* Explanation:

### H4: caching identical inputs
* Question: The same viral image is checked by 50 fact-checkers in one day. Each runs the full pipeline. What is the cache key (content hash of the image/text), what do you store (the fused result), and what is the invalidation policy when you ship a better model?
* Answer:
* Verdict:
* Explanation:

### H5: adding a fifth signal without a rewrite
* Question: A real SynthID client comes online, or you add an audio-deepfake detector. Because every signal is a uniform `{score, available, source}` dict and weights renormalize, what is the minimal change to fold it in? Why does this design make scaling the *number* of signals cheap?
* Answer:
* Verdict:
* Explanation:
