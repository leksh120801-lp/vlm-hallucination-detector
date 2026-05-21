<div align="center">

# VLM Hallucination Detector

### Catch what your vision-language model gets wrong — at the embedding level.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What it does

VLMs (BLIP-2, LLaVA, GPT-4V, etc.) sometimes describe things that aren't there — wrong objects, wrong counts, invented attributes. This tool gives you a **post-hoc alignment score** for any (image, caption) pair without needing access to the original model's internals.

It works by projecting image and caption into a shared embedding space (via CLIP, BLIP, or SigLIP) and computing cosine similarity. Captions below a calibrated threshold are flagged as **possible hallucinations**.

---

## Features

- **Three detection methods** — Threshold (cosine baseline), Consistency (adversarial gap), and Logistic Regression (learned head)
- **Three backbones** — CLIP, BLIP, SigLIP, all behind a unified interface
- **Adversarial evaluation** — auto-generated wrong captions (object swaps + GPT-2) for benchmarking
- **Attention heatmaps** — visualize which image patches the model focused on
- **Streamlit dashboard** — upload images, pick datasets, compare models, view heatmaps
- **FastAPI service** — HTTP endpoint for scoring at scale
- **Datasets included** — Flickr30k, COCO Karpathy, Visual Genome

---

## Detection methods

| Method | How it works | When to use |
|---|---|---|
| **Threshold** | Flags captions where cosine similarity < τ | Fast baseline, no training needed |
| **Consistency** | Flags captions that don't beat adversarial score by τ | More robust to backbone score shifts |
| **Logistic** | Learned head over similarity-score features | Best accuracy on small datasets |

All three methods are selectable in the Streamlit UI and the FastAPI endpoint.

### Backbones

| Model | HuggingFace checkpoint | Notes |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | Default — fastest |
| BLIP | `Salesforce/blip-image-captioning-base` | Captioning-pretrained |
| SigLIP | `google/siglip-base-patch16-224` | Stronger zero-shot |

Thresholds differ per backbone — see `configs/thresholds.yaml`.

---

## Installation

```bash
git clone https://github.com/<your-username>/vlm-hallucination-detector.git
cd vlm-hallucination-detector

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> First run downloads model weights from HuggingFace (~600 MB). Subsequent runs use cache.

**GPU (optional):** If you have CUDA, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before the step above. The model registry auto-detects and moves weights to GPU.

---

## Usage

### Interactive dashboard (recommended)

```bash
streamlit run frontend/streamlit_app.py
```

Upload an image, type a caption, and click Run. Toggle adversarial mode to test against auto-generated wrong captions.

### CLI — score an image against captions

```bash
python main.py
```

```
a dog sitting on grass : 0.314 → likely correct
a cat sitting on grass : 0.198 → possible hallucination
```

### Dataset evaluation

```bash
python experiments/evaluation.py          # COCO + LLM attacks + heatmaps
python experiments/run_benchmark.py       # CLIP vs BLIP vs SigLIP comparison
```

### FastAPI service

```bash
pip install ".[api]"
uvicorn api.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/score \
     -F image=@photo.jpg \
     -F "captions=a cat on grass" \
     -F model=CLIP
```

### Train the logistic detector

```bash
python experiments/train_logistic.py
python experiments/train_logistic.py --per-caption --do-grid-search
```

### Run tests

```bash
pytest -q
```

---

## Configuration

Key settings you may want to change:

| Setting | File | Default |
|---|---|---|
| Decision threshold τ | `configs/thresholds.yaml` | 0.25 (CLIP) |
| Backbone | `main.py` / UI selector | CLIP |
| Eval dataset size | `experiments/evaluation.py` | 10 samples |
| Adversarial count (GPT-2) | `utils/llm_attack.py` | 3 per image |
| Output directory | `experiments/results/` | fixed path |

---

## How it works

```
Image + Caption(s)
       │
       ▼
 Model Registry (CLIP / BLIP / SigLIP)
       │
       ├─ Vision Encoder → Image Embedding ─┐
       └─ Text Encoder  → Text Embedding  ─┤
                                            ▼
                                    Cosine Similarity
                                            │
                              ┌─────────────┴──────────────┐
                          score ≥ τ                    score < τ
                         likely correct           possible hallucination
```

CLIP also produces a **patch-norm heatmap** — showing which parts of the image carry the most signal — without needing gradients or attention weights.

---

## Project structure

```
vlm-hallucination-detector/
├── main.py                    # CLI entry point
├── api/app.py                 # FastAPI service
├── frontend/streamlit_app.py  # Streamlit dashboard
├── models/
│   ├── model_registry.py      # Unified load_model_by_name()
│   ├── clip_model.py
│   ├── blip_model.py
│   └── siglip_model.py
├── utils/
│   ├── similarity.py          # Cosine scoring + decision
│   ├── methods.py             # Threshold / Consistency / Logistic
│   ├── real_heatmap.py        # Patch-norm attention heatmap
│   ├── caption_attack.py      # Adversarial caption generation
│   ├── datasets.py            # Dataset loaders
│   └── metrics.py             # Accuracy, confusion matrix, F1
├── experiments/               # Evaluation scripts + saved results
├── tests/                     # Pytest suite
└── configs/thresholds.yaml    # Per-backbone decision thresholds
```

---

## Known limitations and currently working on

- Thresholds are not auto-calibrated per backbone — SigLIP and BLIP score on different scales than CLIP.
- Eval sets are small by default (designed for fast iteration). Scale up `--sample-size` for meaningful benchmarks.
- Adversarial labels are approximate — an object-swap caption might still be valid if the swapped object is also in the image.
- GPT-2 attacks can produce noisy/echo text; results from LLM-attack runs reflect that noise.
- No batched inference — one image at a time in both the CLI and dashboard.

---

## Contributing

1. Fork and create a branch: `git checkout -b feat/<name>`
2. Run `ruff check` and `pytest -q` before opening a PR
3. Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`

---

## License

MIT — see [LICENSE](LICENSE).
