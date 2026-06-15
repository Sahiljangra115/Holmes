# Y1 Interview Questions

Broader than the per-line walkthrough. What a one-hour ML/backend interview
(L3 to L4 target) asks about a multi-signal AI-content/provenance detector:
calibration, signal fusion, honesty under uncertainty, and the design defense.

Recurring scenario: **Priya**, a newsroom fact-checker, uses your service to
decide whether a suspect article + image is safe to publish. Later it is
licensed to a platform screening millions of uploads a day.

Format: answer in your own words, leave Verdict/Explanation for grading.

---

## Section A: Calibration (the honesty lever)

### A1: Why calibrate at all
* Question: Your classifier's raw softmax is "usually over-confident". Explain what miscalibration means concretely: a model that says 90% but is right only 70% of the time. Why is that worse for Priya (who acts on the number) than a slightly less accurate but honestly-calibrated model?
* Answer:
* Verdict:
* Explanation:

### A2: Temperature scaling specifically
* Question: Of all calibration methods (Platt, isotonic, temperature), you chose temperature scaling. Defend it: why is a single scalar T the right amount of complexity here, what does it preserve (accuracy), and when would you need something richer than one global temperature?
* Answer:
* Verdict:
* Explanation:

### A3: Measuring calibration
* Question: How would you actually prove your confidence is calibrated? Define a reliability diagram and Expected Calibration Error (ECE). What does a well-calibrated curve look like, and what does an over-confident model's curve bow toward?
* Answer:
* Verdict:
* Explanation:

---

## Section B: Signal Fusion

### B1: Why weighted average over a learned fusion model
* Question: You fuse four signals with a fixed-weight transparent average rather than training a meta-classifier on top. Defend the transparent choice for this product (explainability, no fusion training data, graceful degradation). When would a learned fusion (stacking) win?
* Answer:
* Verdict:
* Explanation:

### B2: Graceful degradation
* Question: The renormalize-over-available-signals trick means a missing signal cannot move the score. Explain why this is more honest than imputing a default, and walk the case where only text is available: what does Priya's confidence reduce to?
* Answer:
* Verdict:
* Explanation:

### B3: Why these weights
* Question: text=0.45, image=0.40, metadata=0.10, watermark=0.05. An interviewer asks "where did these come from, did you tune them?" Give the honest answer (hand-set priors reflecting signal trustworthiness) and how you would learn them properly if you had labeled fused outcomes.
* Answer:
* Verdict:
* Explanation:

### B4: The uncertain band as a product decision
* Question: Inputs between 0.35 and 0.65 land in "uncertain". Why is refusing to call it sometimes the right product behavior for a fact-checking tool? What is the cost of forcing a binary call on a genuinely ambiguous input?
* Answer:
* Verdict:
* Explanation:

---

## Section C: The LLM's Role

### C1: Narrator, not judge
* Question: The LLM in `explain` only narrates numbers the code already computed; the band is decided by deterministic thresholds. Defend this separation. What category of failure (hallucinated verdict, prompt injection flipping the call) does it structurally prevent?
* Answer:
* Verdict:
* Explanation:

### C2: Grounding the explanation
* Question: The prompt feeds pre-computed numbers and says "do not invent numbers". Why is this a form of grounding, and how does it relate to the RAG idea of constraining a model to provided context? What can still go wrong even with this constraint?
* Answer:
* Verdict:
* Explanation:

### C3: Deterministic fallback
* Question: When Ollama is unreachable, `explain` returns a template. Argue why a deterministic, slightly worse explanation beats a flaky LLM call for a tool Priya relies on under deadline. What does this say about treating the LLM as optional polish, not core logic?
* Answer:
* Verdict:
* Explanation:

---

## Section D: Provenance and Adversaries

### D1: C2PA's promise and limits
* Question: C2PA is cryptographically signed provenance. Why does its presence only nudge toward authentic (weight 0.10) rather than prove it? Conversely, why is its absence not evidence of forgery? Name the two asymmetries.
* Answer:
* Verdict:
* Explanation:

### D2: Metadata is forgeable
* Question: An adversary fakes EXIF to mimic a real camera and strips any AI watermark. Which of your four signals does this defeat, and which still has a chance? Why is a multi-signal design more robust to one forged channel than a single-signal detector?
* Answer:
* Verdict:
* Explanation:

### D3: The arms race
* Question: Generators improve to evade your image detector. Your detector's accuracy decays over months. Is this a bug or the nature of the problem? How does honest calibration and the "estimate, not oracle" framing keep the product useful even as raw accuracy drifts?
* Answer:
* Verdict:
* Explanation:

---

## Section E: The Text Model

### E1: Why DeBERTa-v3-base
* Question: Defend DeBERTa-v3-base for the LIAR2 credibility task over a smaller model (DistilBERT) or a much larger one. Tie it to the accuracy/latency/GPU-budget tradeoff and the fact that this is one of four signals, not the whole product.
* Answer:
* Verdict:
* Explanation:

### E2: The binary collapse is a judgment call
* Question: Collapsing 6-way LIAR2 truth into credible/not_credible loses information. Defend the collapse for this product, name what you lose (half-true vs barely-true distinction), and why stating the mapping in the README is non-negotiable for the metric to be honest.
* Answer:
* Verdict:
* Explanation:

### E3: Credibility is not truth
* Question: Your model predicts "credible", not "true". Explain the gap an interviewer is probing: a credible-sounding false statement vs an awkwardly-worded true one. Why must the product never claim to detect truth, only a credibility signal?
* Answer:
* Verdict:
* Explanation:

---

## Section F: System Design

### F1: The full fuse request path
* Question: Walk `/predict/fuse` from request to response: text predict, image predict, metadata, watermark, fuse, explain. Where is the latency, which calls are parallelizable, and which would you cut first if the endpoint felt slow?
* Answer:
* Verdict:
* Explanation:

### F2: Lazy loading and cold starts
* Question: Models load lazily and cache per container. Explain the cold-start tax on Lambda and why the lazy pattern is the right tradeoff for a multi-endpoint API where most requests touch only one model. When does lazy loading stop being enough?
* Answer:
* Verdict:
* Explanation:

### F3: The uniform signal contract
* Question: Every signal is `{score, available, source}`. Why is this uniform shape the key design decision that makes the system extensible and testable? How would you unit-test fusion without running any real model?
* Answer:
* Verdict:
* Explanation:

---

## Section G: The "Why" Questions (architecture defense)

### G1: Why multi-signal over one big model
* Question: Why not train one end-to-end model that eats text+image+metadata and outputs a verdict? Give the honest reasons (per-signal calibration, graceful degradation, explainability, no fused training data) an interviewer wants, and the case where end-to-end would win.
* Answer:
* Verdict:
* Explanation:

### G2: Why "estimate, not oracle"
* Question: The whole product is built around never claiming certainty. An interviewer says "users want a clear fake/real verdict, your uncertainty is a cop-out." Defend the design as appropriate to the actual difficulty of the problem and to a fact-checker's real workflow.
* Answer:
* Verdict:
* Explanation:

### G3: What you would cut under a deadline
* Question: One week instead of full scope. Which signal do you ship first and which do you stub (the SynthID UNAVAILABLE pattern is a hint)? Why is the text-only calibrated slice a defensible tracer bullet?
* Answer:
* Verdict:
* Explanation:

---

# Reliability and Scalability (appended)

Production-grade questions: failure modes, load, and what an SRE would ask.

## Section H: Reliability

### H1: One bad signal poisons the average
* Question: Fusion is a weighted mean, so a miscalibrated image detector drags the fused confidence. How do you detect a drifting signal in production (per-signal calibration monitoring), and would you auto-drop a signal whose calibration degrades past a threshold?
* Answer:
* Verdict:
* Explanation:

### H2: Graceful capability reporting
* Question: On a deploy where the text model is missing but image works, `/predict/fuse` partially degrades via the available-signals logic, but `/predict/text` fails hard. How would `/health` advertise which signals are live so Priya's UI can grey out unavailable ones instead of erroring mid-request?
* Answer:
* Verdict:
* Explanation:

### H3: Bounding the LLM and external calls
* Question: `explain` calls Ollama; a real SynthID client would call an external API. Describe the timeout, retry-with-backoff, and circuit-breaker policy so a slow upstream cannot pile up Lambda concurrency. Why is narration safe to make best-effort while a verdict must not be?
* Answer:
* Verdict:
* Explanation:

### H4: Observability for a trust product
* Question: A trust tool that is silently wrong is dangerous. Name the three things you would instrument first (per-signal availability rate, band distribution over time, calibration error on labeled spot-checks) and what each early-warns you about.
* Answer:
* Verdict:
* Explanation:

### H5: Reproducibility and audit
* Question: Priya publishes based on your "likely_authentic" call and is later challenged. Can you reproduce exactly what the system said that day (model version, weights, temperature, input hash)? What do you log to make every verdict auditable?
* Answer:
* Verdict:
* Explanation:

## Section I: Scalability

### I1: Platform-scale screening
* Question: Licensed to screen millions of uploads/day. Walk every layer that changes: model serving (Lambda to GPU batch workers), fusion (stays cheap), watermark (external API rate limits), and the LLM narration (probably dropped or async at this scale). Which is the first bottleneck?
* Answer:
* Verdict:
* Explanation:

### I2: Batching and ONNX
* Question: The sibling y2 project exports its model to ONNX for cheap CPU serving. Make the case for doing the same to DeBERTa here, plus dynamic batching of the text forward pass. What accuracy/latency tradeoff does quantization introduce?
* Answer:
* Verdict:
* Explanation:

### I3: Caching viral content
* Question: The same deepfake image is submitted thousands of times in an hour. What is the cache key (content hash), what do you cache (the full fused result and explanation), and how do you invalidate when you ship an improved detector without serving stale verdicts forever?
* Answer:
* Verdict:
* Explanation:

### I4: Cost asymmetry across signals
* Question: Metadata parsing is microseconds; two model forward passes and an LLM call are the cost. At platform scale, how do you order the pipeline to fail fast and cheap (run cheap signals first, short-circuit obvious cases) before paying for the expensive ones?
* Answer:
* Verdict:
* Explanation:

### I5: Scaling the number of signals
* Question: You want to add audio-deepfake and reverse-image-search signals. Because of the uniform signal contract and renormalizing weights, what is the marginal cost of a new signal versus a system where fusion was a hard-coded formula? Why does this design make horizontal signal growth cheap?
* Answer:
* Verdict:
* Explanation:

### I6: Multi-region and latency
* Question: Newsrooms are global but your models sit in one region. What breaks for a Tokyo fact-checker on latency, and what is the minimal change (regional model replicas, edge caching of results) to keep the tool fast without duplicating the whole stack everywhere?
* Answer:
* Verdict:
* Explanation:
