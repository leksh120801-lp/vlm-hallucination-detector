# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- _placeholder for the next release_

## [0.2.0] - 2026-04-29

### Added
- **Three pluggable detection methods** (Threshold / Consistency / Logistic), unified behind `utils.methods`. The Streamlit UI and FastAPI service both expose the picker.
- **Logistic-regression detector** trained over similarity-score *statistics* (4-D or 8-D), much more sample-efficient than embedding-feature heads. New modules:
  - `utils/classifier.py` — feature engineering + joblib persistence.
  - `experiments/train_logistic.py` — sklearn `Pipeline` (`StandardScaler` + `LogisticRegression`) with optional grid search, single-class fallback to `DummyClassifier`, and per-image / per-caption variants.
- **Cross-method evaluation script** `experiments/evaluate_methods.py` — runs Threshold / Consistency / Logistic side-by-side and writes a comparison CSV.
- **`compute_classification_metrics`** in `utils/metrics.py` — sklearn-based accuracy / precision / recall / F1 against `y_true` / `y_pred` lists.
- **`utils/insights.py`** — best-method-by-F1 summarizer and improvement-over-baseline helper used by the Streamlit dashboard.
- **Centralised logging** via `utils/logging_config.py` — replaces every `print(...)` debug statement; level is configurable through the `VLMHALL_LOG_LEVEL` env var.
- New tests: `tests/test_classifier.py`, `tests/test_classifier_features.py`, `tests/test_train_logistic.py`.

### Changed
- **Datasets are now properly working again.** `nlphuji/flickr30k` and `HuggingFaceM4/NoCaps` use the embedded PIL image (no URL fetching) — the previous URL-based path was broken on `datasets >= 2.14`. COCO Karpathy still fetches by URL with timeout + retry-skip. `requirements.txt` and `pyproject.toml` pin `datasets>=2.14,<3.0`.
- **Heatmap is now unified across CLIP / BLIP / SigLIP** via `generate_heatmap(model, processor, image, caption, model_name=...)`. The legacy `generate_clip_heatmap` is preserved as a backwards-compatible alias.
- **Streamlit app rewritten** — three-method picker, per-(model, method) decision cards, per-caption aggregate metrics with `compute_classification_metrics`, "Summary Insight" panel, adversarial breakdown table, and side-by-side multi-model heatmaps. Cleaned up CSS bloat.
- **FastAPI service** now exposes the three detection methods via `POST /v1/score?method=…`. Pydantic schemas updated.
- `experiments/evaluation.py` and `experiments/run_benchmark.py` now use argparse + the centralised logger and write into `experiments/results/` reproducibly.
- `main.py` is now a proper CLI: accepts `--image`, `--captions`, `--threshold`.

### Removed (deprecated stubs)
- `models/verification_head.py` — superseded by `utils/classifier.py` + `utils/methods.py`. Stub raises `ImportError`.
- `experiments/train_verification_head.py` — superseded by `experiments/train_logistic.py`. Stub exits with a redirect message.
- `tests/test_verification_head.py` — replaced by `tests/test_classifier*.py`. Marked as `pytest.mark.skip`.
- `experiments/benchmark_table.py` — superseded by `experiments/evaluate_methods.py`. Stub exits with a redirect message.
- (Cleanup) Run `git rm` on the four stubs above to remove them entirely.

### Fixed
- (carried over from v0.1.0) Inner-loop scope bug in `experiments/evaluation.py`.
- (carried over from v0.1.0) Decision-string mismatch in `utils/metrics.py`.

## [0.1.0] - 2026-04-28

### Added
- Multi-backbone hallucination detector with a unified `model_registry` over CLIP, BLIP, and SigLIP.
- Cosine-similarity scoring with per-backbone configurable thresholds (`configs/thresholds.yaml` + `utils/config.py`).
- Adversarial caption generators: rule-based object swaps and GPT-2-prompted distractors.
- CLIP patch-norm attention heatmap with overlay rendering.
- Streamlit dashboard with glassmorphism UI.
- FastAPI service skeleton with `/health` and `POST /v1/score` endpoints.
- Multi-stage CPU `Dockerfile` with healthcheck, plus `.dockerignore` and `docker-compose.yml`.
- Pytest suite covering similarity, metrics, attacks, config (offline-safe).
- GitHub Actions CI running ruff and pytest on Python 3.10 and 3.11.
- `pyproject.toml` with `[api]`, `[ui]`, `[dev]`, `[all]` extras.
- MIT license.
- `IMPROVEMENTS.md` (Amazon-grade critique) and `PACKAGING.md` (library-conversion guide).

### Fixed
- Inner-loop scope bug in `experiments/evaluation.py`.
- Decision-string mismatch in `utils/metrics.py`.
