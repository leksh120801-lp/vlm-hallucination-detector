# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- _placeholder for the next release_

## [1.0.0] - 2026-04-29

First public release.

### Detection
- Three pluggable detection methods unified behind `utils.methods`:
  - **Threshold** — cosine similarity ≥ τ baseline.
  - **Consistency** — original score must beat the mean adversarial score by τ.
  - **Logistic** — learned head over similarity-score statistics
    (`utils.classifier`, trained via `experiments/train_logistic.py`).
- Per-backbone decision thresholds in `configs/thresholds.yaml`.

### Backbones
- Multi-backbone support: CLIP, BLIP, SigLIP, accessed through a thread-safe
  three-tier cache (`models/model_registry.py`): in-process singleton →
  optional joblib disk cache → cold load.
- BLIP loader now uses `BlipForConditionalGeneration` with an in-file adapter
  that exposes CLIP-style `image_embeds` / `text_embeds`, eliminating
  deprecation + missing-weight warnings.
- `HF_TOKEN` honoured (anonymous access still works when unset).
- `use_fast=True` passed to image processors to silence migration warnings.

### Datasets
- Two dataset loaders, all caching handled by HuggingFace's standard cache
  (`HF_HOME`):
  - `nlphuji/flickr30k` — embedded PIL images.
  - `yerevann/coco-karpathy` — image URLs cast to `datasets.Image()` for
    lazy fetch + cache, no manual HTTP from the codebase.
- Schema-tolerant caption extractor handling both the `sentences` and
  `caption` field shapes.

### Evaluation
- `experiments/evaluate_methods.py` — three-method comparison producing a CSV
  at `experiments/results/method_comparison.csv`, with backbone-mismatch
  warning when the trained logistic head's backbone differs from evaluation.
- `experiments/evaluate_pope.py` — POPE benchmark adapter (random / popular /
  adversarial splits), reporting accuracy / precision / recall / F1.
- `experiments/run_benchmark.py` — Flickr30k cosine-baseline comparison across
  CLIP / BLIP / SigLIP.
- `utils/metrics.py` — accuracy + hallucination rate, confusion matrix with
  derived precision / recall / F1, and `compute_classification_metrics`
  for sklearn-style scoring.

### Adversarial harness
- `utils/caption_attack.py` — rule-based object-swap perturbations + hard
  distractors.
- `utils/llm_attack.py` — GPT-2-prompted adversarial captions for the
  evaluation script.

### Interfaces
- **Streamlit dashboard** (`frontend/streamlit_app.py`) — input modes
  (upload + dataset sample), per-(model, method) decision cards, sorted
  comparison table, attention heatmap strip, adversarial breakdown.
- **Summary Insight panel** with two display modes:
  - *single config* (one model + one method) — direct F1 / Precision / Recall /
    Accuracy tiles plus interpretation badge, no "best vs others" framing.
  - *comparison* (any "ALL" selection) — best configuration headline + sorted
    comparison table + lift over the runner-up.
- **FastAPI service** (`api/app.py`) — `GET /health`, `POST /v1/score?method=…`
  with Pydantic schemas and lazy backbone caching.
- `main.py` CLI accepts `--image`, `--captions`, `--threshold`.

### Visualisation
- Unified patch-norm attention heatmap for CLIP / BLIP / SigLIP via
  `generate_heatmap(model, processor, image, caption, model_name=...)`.
  CLIP-specific helper preserved as a backwards-compatible alias.

### Quality
- Centralised logger (`utils/logging_config.py`) replaces every `print(...)`
  debug call. Level configurable via `VLMHALL_LOG_LEVEL`.
- Strengthened HuggingFace logging suppression: `transformers`,
  `huggingface_hub`, and `FutureWarning` / `UserWarning` from the relevant
  modules. Only meaningful errors reach the terminal.
- Pytest suite covering similarity, metrics, attacks, classifier, config,
  and the logistic training pipeline (offline-safe).
- GitHub Actions CI — ruff + pytest on Python 3.10 and 3.11.
- Multi-stage CPU `Dockerfile` + `docker-compose.yml`.
- `pyproject.toml` with `[api]`, `[ui]`, `[dev]`, `[all]` extras.
- MIT license.

### Documentation
- README with capability section ("Designing experiments and statistical
  analysis", "Implementing algorithms using toolkits and self-developed code",
  "Solving business problems through ML"), usage modes, and a real-world
  vignette.
- Architecture diagrams (Mermaid) and module-responsibilities table in
  `docs/architecture.md`.
- `IMPROVEMENTS.md` and `PACKAGING.md` for future direction.
