# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (placeholder for the next release)

## [0.1.0] - 2026-04-28

### Added
- Multi-backbone hallucination detector with a unified `model_registry` over CLIP, BLIP, and SigLIP.
- Cosine-similarity scoring with per-backbone configurable thresholds (`configs/thresholds.yaml` + `utils/config.py`).
- Adversarial caption generators: rule-based object swaps (`utils/caption_attack.py`) and GPT-2-prompted distractors (`utils/llm_attack.py`).
- CLIP patch-norm attention heatmap (`utils/real_heatmap.py`) with overlay rendering.
- Streamlit dashboard with glassmorphism UI: upload mode, dataset-sample mode, model comparison cards, attention overlay (`frontend/streamlit_app.py`).
- FastAPI service with `/health`, `/`, `POST /v1/score` endpoints, Pydantic schemas, and an in-process model cache (`api/app.py`).
- Multi-stage CPU `Dockerfile` with healthcheck, plus `.dockerignore`.
- Pytest suite covering `utils.similarity`, `utils.metrics`, `utils.caption_attack`, `utils.config` (~22 tests, offline-safe).
- GitHub Actions CI (`.github/workflows/ci.yml`) running ruff and pytest on Python 3.10 and 3.11.
- `pyproject.toml` with `[api]`, `[ui]`, `[dev]`, `[all]` extras.
- `experiments/benchmark_table.py` — generates a markdown performance table per backbone.
- Defensive dataset loaders (`utils/datasets.py`) with curated Wikimedia Commons fallback when HuggingFace dataset scripts aren't available.
- MIT license.

### Fixed
- Inner-loop scope bug in `experiments/evaluation.py` that caused only the last caption per image to be recorded.
- Decision-string mismatch in `utils/metrics.py` (`"hallucination"` vs `"possible hallucination"`) that left the hallucination counter permanently at zero. Replaced literal strings with module-level constants and added derived `precision` / `recall` / `f1`.

### Documentation
- Full professional README with architecture diagram, installation, multi-mode usage, project structure, and a real-world vignette.
- `IMPROVEMENTS.md` — Amazon-grade critique with a 14-item readiness checklist.
- `PACKAGING.md` — end-to-end guide for converting the project into a `pip install`-able package.
