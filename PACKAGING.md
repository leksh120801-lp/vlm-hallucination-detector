# Turning This Project Into a Reusable Python Package

> Goal: take the existing repo and ship it as `vlmhall` — a `pip install`-able
> library with a CLI, an importable Python API, tests, docs, CI, and (optionally)
> a PyPI release. The `pyproject.toml`, `LICENSE`, tests, FastAPI service,
> Dockerfile, and CI in this repo already cover most of the foundations; this
> document explains the missing layout work and the release flow end to end.

---

## 1. Restructure: From Script Repo → Library Layout

The current layout is "script-first" — top-level `models/`, `utils/`, `main.py`.
For a published package, the recommended convention is the **`src/` layout**,
because it prevents accidental imports of un-installed code (the single most
common source of "works on my machine" bugs in published packages).

### Target tree

```
vlm-hallucination-detector/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .gitignore
├── .github/workflows/ci.yml
│
├── src/
│   └── vlmhall/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── types.py
│       ├── exceptions.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py            # Backbone ABC
│       │   ├── clip.py
│       │   ├── blip.py
│       │   ├── siglip.py
│       │   └── registry.py
│       ├── detectors/
│       │   ├── __init__.py
│       │   ├── base.py            # Detector ABC
│       │   ├── cosine.py
│       │   └── verification_head.py
│       ├── attacks/
│       │   ├── object_swap.py
│       │   └── llm_attack.py
│       ├── data/
│       │   ├── flickr.py
│       │   ├── coco.py
│       │   └── visual_genome.py
│       ├── viz/
│       │   ├── heatmap.py
│       │   └── plots.py
│       ├── eval/
│       │   ├── metrics.py
│       │   └── runner.py
│       └── service/
│           └── api.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_similarity.py
│   │   ├── test_metrics.py
│   │   ├── test_attacks.py
│   │   └── test_config.py
│   └── integration/
│       └── test_end_to_end.py
│
├── examples/
│   ├── single_image.py
│   └── benchmark.py
│
├── docs/
│   ├── index.md
│   ├── architecture.md
│   └── tutorials/
│       └── quickstart.md
│
└── frontend/
    └── streamlit_app.py
```

### Refactor moves (one-to-one)

| From | To |
|---|---|
| `models/clip_model.py:load_clip_model` | `vlmhall.models.clip:CLIPBackbone` |
| `models/blip_model.py` | `vlmhall.models.blip` |
| `models/siglip_model.py` | `vlmhall.models.siglip` |
| `models/model_registry.py` | `vlmhall.models.registry` |
| `utils/similarity.py` | `vlmhall.detectors.cosine` |
| `utils/preprocessing.py` | `vlmhall.data.io` |
| `utils/visualization.py` + `utils/real_heatmap.py` | `vlmhall.viz.heatmap` |
| `utils/caption_attack.py` | `vlmhall.attacks.object_swap` |
| `utils/llm_attack.py` | `vlmhall.attacks.llm_attack` |
| `utils/datasets.py` | `vlmhall.data.{flickr,coco,visual_genome}` |
| `utils/metrics.py` | `vlmhall.eval.metrics` |
| `utils/plots.py` | `vlmhall.viz.plots` |
| `utils/config.py` | `vlmhall.config` |
| `experiments/evaluation.py` | `vlmhall.eval.runner` + `examples/benchmark.py` |
| `main.py` | `vlmhall.cli:main` |
| `api/app.py` | `vlmhall.service.api:app` |

---

## 2. `pyproject.toml`

Already in this repo. The metadata block (`name`, `version`, `dependencies`,
`[project.optional-dependencies]`, `[project.scripts]`) is the modern standard
and replaces `setup.py`. Keep tool config (`[tool.ruff]`, `[tool.pytest.ini_options]`)
in the same file.

After the `src/` move, update the `setuptools` block:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

…and add a CLI entry point so `vlmhall ...` works after install:

```toml
[project.scripts]
vlmhall = "vlmhall.cli:app"
```

---

## 3. `requirements.txt` and lockfiles

`pyproject.toml` describes *abstract* version ranges; `requirements.txt` is a
*concrete* lock for reproducible installs. Keep both. To regenerate:

```bash
pip install pip-tools
pip-compile pyproject.toml -o requirements.txt
pip-compile --extra dev pyproject.toml -o requirements-dev.txt
```

Or with the newer `uv`:

```bash
uv pip compile pyproject.toml -o requirements.txt
```

Commit both. Bump them on every dependency change.

---

## 4. Versioning (Semantic Versioning)

Adopt **SemVer 2.0.0**: `MAJOR.MINOR.PATCH`.

- `MAJOR` — breaking API change.
- `MINOR` — backwards-compatible feature.
- `PATCH` — backwards-compatible fix.
- Pre-1.0 (`0.x.y`): the API is allowed to break on minor bumps. Say so in the README.

Store the version **once**, in `pyproject.toml`, and read it at runtime:

```python
# src/vlmhall/__init__.py
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("vlmhall")
except PackageNotFoundError:
    __version__ = "0.0.0+local"
```

Maintain a `CHANGELOG.md` (Keep a Changelog format) and update it on every release.

---

## 5. Docstrings & API Docs

Use **Google-style docstrings**; render them with **MkDocs Material + mkdocstrings**.

```python
def cosine_similarity(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between image and text embeddings.

    Both inputs are L2-normalized along the last dimension, then the dot
    product is returned. Output values lie in `[-1, 1]`.

    Args:
        image_embeds: Tensor of shape `(B_img, D)`.
        text_embeds:  Tensor of shape `(B_txt, D)`.

    Returns:
        A `(B_img, B_txt)` tensor of pairwise cosine similarities.
    """
    ...
```

Build & deploy:

```bash
pip install ".[docs]"
mkdocs serve            # local preview at :8000
mkdocs gh-deploy        # publishes to gh-pages
```

---

## 6. CLI Support (Typer + Rich)

```python
# src/vlmhall/cli.py
import typer
from rich import print

app = typer.Typer(no_args_is_help=True)

@app.command()
def score(image: str, caption: str, model: str = "CLIP", threshold: float = 0.25):
    """Score one image–caption pair."""
    ...

@app.command()
def benchmark(dataset: str = "flickr", n: int = 30, model: str = "CLIP", out: str = "results.json"):
    """Run an adversarial benchmark."""
    ...
```

After install:

```bash
vlmhall score path/to/cat.jpg "a cat on grass" -m SigLIP -t 0.22
vlmhall benchmark --dataset coco --n 100 --model CLIP --out coco.json
```

---

## 7. Unit Tests

Already in `tests/` (post-refactor: `tests/unit/`). Aim for ≥ 80 % coverage on
`detectors`, `eval.metrics`, `models.registry`, `config`. Mock model loaders so
tests don't download weights.

```bash
pytest -q
pytest --cov=vlmhall --cov-report=term-missing
```

---

## 8. Build

```bash
pip install build
python -m build           # creates dist/vlmhall-0.1.0.tar.gz + dist/*.whl
```

Verify before publishing:

```bash
pip install twine
twine check dist/*
python -m venv /tmp/check && /tmp/check/bin/pip install dist/vlmhall-0.1.0-py3-none-any.whl
/tmp/check/bin/vlmhall --help
```

---

## 9. Publishing

### A. GitHub release (always)

1. Tag: `git tag v0.1.0 && git push --tags`.
2. Draft a release on GitHub from the tag; paste the `CHANGELOG.md` section.
3. Attach the wheel + sdist from `dist/` (or let CI do it — see release workflow below).

```yaml
# .github/workflows/release.yml
name: release
on:
  push:
    tags: ["v*"]
jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions: { contents: write, id-token: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install build
      - run: python -m build
      - uses: softprops/action-gh-release@v2
        with: { files: "dist/*" }
      - uses: pypa/gh-action-pypi-publish@release/v1   # OIDC, no API token
```

### B. PyPI (optional)

Use **Trusted Publishers** (OIDC) so you never store a PyPI token in GitHub.

1. Register the project on https://pypi.org.
2. In *Project → Settings → Publishing*, add a Trusted Publisher: GitHub, repo, workflow, environment.
3. Push a tag — the workflow above publishes automatically.

Manual fallback:

```bash
twine upload dist/*
```

### C. TestPyPI first (always)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ vlmhall
```

---

## 10. How Users Install and Use It

```bash
pip install vlmhall                 # core
pip install "vlmhall[api]"          # with FastAPI service
pip install "vlmhall[ui]"           # with Streamlit UI
pip install "vlmhall[all]"          # everything
```

```bash
vlmhall score ./cat.jpg "a cat on grass"
vlmhall benchmark --dataset coco --n 100 --out coco.json

uvicorn vlmhall.service.api:app --port 8000
```

```python
from PIL import Image
from vlmhall.models.registry import get_backbone
from vlmhall.detectors.cosine import CosineDetector

backbone = get_backbone("SigLIP")
detector = CosineDetector(threshold=0.22)
results = detector.score(backbone, Image.open("cat.jpg"), [
    "a cat on grass",
    "a dinosaur in a city",
])
for r in results:
    print(f"{r.caption:40s}  {r.score:.3f}  → {r.decision}")
```

---

## 11. Migration Checklist (in order)

1. Create `src/vlmhall/` and move files (table in §1).
2. Update `pyproject.toml`: `tool.setuptools.packages.find.where = ["src"]`, add `[project.scripts]`.
3. Replace bare `try/except` with typed exceptions in `vlmhall.exceptions`.
4. Add type hints + Google docstrings everywhere public.
5. Add Pydantic models in `vlmhall.types` for `ScoreResult`, `Sample`, `EvalRun`.
6. Implement Detector ABC; rewrite cosine path against it.
7. Add Typer CLI; register entry point.
8. Move tests to `tests/unit/`; add `tests/integration/`.
9. Add `mkdocs.yml`; deploy docs to GitHub Pages.
10. Tag `v0.1.0`, build, publish to TestPyPI, smoke-test, then PyPI.

When all 10 are done, the project is no longer "interesting repo" — it's a tool other people install.
