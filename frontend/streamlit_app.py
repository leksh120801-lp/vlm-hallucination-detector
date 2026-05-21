"""Streamlit dashboard for the VLM Hallucination Detector.

Sections (top to bottom):
    1. Inputs        — image upload OR dataset sample, model picker, method picker.
    2. Run controls  — adversarial test toggle, heatmap toggle, run button.
    3. Results       — per-(model, method) decision cards, summary table, bar chart.
    4. Insight       — best method by F1 plus a one-line comparison summary.
    5. Adversarial   — per-attack score table and mini bar chart (when enabled).
    6. Heatmaps      — patch-norm activation maps for CLIP / BLIP / SigLIP.

All ML logic delegates to ``utils.methods`` and ``utils.classifier`` so this
file stays a thin presentation layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `from utils...` and `from models...` work when the
# app is launched from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from models.model_registry import load_model_by_name  # noqa: E402
from utils.caption_attack import generate_adversarial_captions  # noqa: E402
from utils.classifier import load_logistic_model  # noqa: E402
from utils.datasets import (  # noqa: E402
    load_coco_sample,
    load_flickr_sample,
)
from utils.methods import (  # noqa: E402
    HALLUCINATION,
    LIKELY_CORRECT,
    method_consistency_threshold,
    method_logistic,
    method_threshold,
    split_similarity_scores,
)
from utils.metrics import compute_classification_metrics  # noqa: E402
from utils.preprocessing import pil_to_cv2  # noqa: E402
from utils.real_heatmap import generate_heatmap  # noqa: E402
from utils.similarity import compute_similarity  # noqa: E402

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VLM Hallucination Detector",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

[data-testid="stAppViewContainer"] {
    background: #020510;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(88, 28, 235, 0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0, 200, 255, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 60% 30%, rgba(255, 50, 180, 0.12) 0%, transparent 55%);
    animation: bgPulse 10s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { background-position: 0% 0%, 100% 100%, 60% 30%; }
    100% { background-position: 5% 8%, 95% 92%, 65% 35%; }
}
#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: #e2e8f0;
}
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 3rem !important;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #c4b5fd, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
div[data-testid="stExpander"], div.stForm, div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
}
div.stButton > button {
    background: linear-gradient(135deg, #6d28d9 0%, #0369a1 50%, #6d28d9 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer !important;
    transition: background-position 0.6s ease, box-shadow 0.4s ease, transform 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(109, 40, 217, 0.45) !important;
}
div.stButton > button:hover {
    background-position: right center !important;
    box-shadow: 0 6px 36px rgba(109, 40, 217, 0.65) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5) !important;
}
[data-testid="stMarkdownContainer"] p {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stVerticalBlock"] { animation: fadeSlideUp 0.55s ease both; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header banner
# ---------------------------------------------------------------------------

st.markdown(
    """
<div style="
    background: linear-gradient(135deg, rgba(109,40,217,0.18) 0%, rgba(14,165,233,0.12) 100%);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 24px;
    padding: 2.5rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
">
  <div style="display:flex; align-items:center; gap:16px; margin-bottom:0.6rem;">
    <span style="font-size:2.4rem;">🔬</span>
    <h1 style="margin:0; padding:0; font-size:2.2rem;">VLM Hallucination Detector</h1>
  </div>
  <p style="margin:0; font-family:'DM Mono',monospace; font-size:0.88rem; color:#94a3b8;">
    Evaluate vision–language model captions across CLIP · BLIP · SigLIP, with three pluggable detection methods.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_header(title: str, grad_a: str = "#7c3aed", grad_b: str = "#38bdf8") -> None:
    st.markdown(
        f"""
    <div style="margin: 2rem 0 1rem; display:flex; align-items:center; gap:12px;">
      <div style="width:4px; height:28px; border-radius:2px;
        background: linear-gradient(180deg,{grad_a},{grad_b});"></div>
      <h2 style="margin:0; font-size:1.25rem;">{title}</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _render_heatmap_overlay(image_cv: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Resize, gamma-correct, colour-map and blend a heatmap onto the image."""
    h, w = image_cv.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = hm / (hm.max() + 1e-8)
    hm = hm ** 3
    hm_u8 = np.uint8(255 * hm)
    coloured = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image_cv, 0.5, coloured, 0.7, 0)


@st.cache_data(show_spinner=False)
def _load_samples_cached(choice: str, n: int = 10):
    """Cache dataset samples per choice so the slider doesn't re-trigger downloads."""
    if choice == "Flickr30k":
        return load_flickr_sample(n)
    return load_coco_sample(n)


@st.cache_resource(show_spinner=False)
def _load_logistic_model_cached():
    """Lazy-load the logistic model. Returns ``None`` if not yet trained."""
    try:
        return load_logistic_model()
    except (FileNotFoundError, OSError):
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

col_mode, col_model, col_method = st.columns([1, 1, 1], gap="large")

with col_mode:
    st.caption("Input Mode")
    mode = st.radio(
        "Input Mode",
        ["Upload Image", "Dataset Sample"],
        label_visibility="collapsed",
    )

with col_model:
    st.caption("Model")
    model_choice = st.selectbox(
        "Model",
        ["CLIP", "BLIP", "SigLIP", "ALL"],
        label_visibility="collapsed",
    )

with col_method:
    st.caption("Method")
    method_choice = st.selectbox(
        "Method",
        ["Threshold", "Consistency", "Logistic", "ALL"],
        label_visibility="collapsed",
        help=(
            "Threshold: cosine ≥ τ → correct.\n"
            "Consistency: original score must beat mean adversarial score.\n"
            "Logistic: learned head over similarity-score statistics "
            "(train it with `python experiments/train_logistic.py`)."
        ),
    )

st.markdown("---")

image: Image.Image | None = None
caption: str | None = None

if mode == "Upload Image":
    col_up, col_cap = st.columns([1, 1], gap="large")

    with col_up:
        st.caption("Image")
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, width="stretch")

    with col_cap:
        st.caption("Caption")
        caption = st.text_input(
            "Enter caption",
            label_visibility="collapsed",
            placeholder="Describe what you see in the image…",
        )

else:
    col_ds, col_slide = st.columns([1, 2], gap="large")

    with col_ds:
        st.caption("Dataset")
        dataset_choice = st.selectbox(
            "Dataset",
            ["Flickr30k", "COCO"],
            label_visibility="collapsed",
        )

        with st.spinner("Loading dataset samples…"):
            try:
                samples = _load_samples_cached(dataset_choice, 10)
            except ModuleNotFoundError:
                st.error(
                    "The `datasets` package is required for dataset mode.\n"
                    "Install with: `pip install 'datasets>=2.14,<3.0'`."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Failed to load samples: {exc}")
                st.stop()

    if not samples:
        st.error(
            "⚠️ Could not load dataset samples. Check your internet connection or "
            "try a different dataset."
        )
        st.stop()

    with col_slide:
        st.caption("Sample Index")
        idx = st.slider(
            "Select sample",
            0,
            len(samples) - 1,
            label_visibility="collapsed",
        )

    image = samples[idx]["image"]
    caption = samples[idx]["caption"]

    col_img, col_cap = st.columns([1, 1], gap="large")
    with col_img:
        st.image(image, width="stretch")
    with col_cap:
        st.markdown(
            f"""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
            border-radius:16px; padding:1.4rem 1.6rem; backdrop-filter:blur(12px);">
          <p style="font-size:0.7rem; color:#7c3aed; text-transform:uppercase;
            letter-spacing:0.12em; margin:0 0 8px; font-weight:700;">Caption</p>
          <p style="font-family:'DM Mono',monospace; font-size:0.9rem; color:#e2e8f0;
            line-height:1.7; margin:0;">{caption}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------

st.markdown("---")
col_opt, col_btn = st.columns([2, 1], gap="large")

with col_opt:
    attack_mode = st.checkbox("⚔️ Run adversarial caption test", value=False)
    show_heatmaps = st.checkbox("🌡️ Show attention heatmaps", value=True)

with col_btn:
    run_button = st.button("⚡ Run Evaluation", width="stretch")


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

if run_button and image and caption:

    models_to_run = ["CLIP", "BLIP", "SigLIP"] if model_choice == "ALL" else [model_choice]

    # Load logistic model if needed.
    logistic_model = None
    if method_choice in {"Logistic", "ALL"}:
        logistic_model = _load_logistic_model_cached()
        if logistic_model is None:
            st.warning(
                "No trained logistic model found at `models/logistic_model.pkl`. "
                "Train one with `python experiments/train_logistic.py`. "
                "The Logistic method will be skipped for this run."
            )

    _section_header("Model Comparison")

    progress = st.progress(0)
    status = st.empty()

    summary_rows: list[dict] = []          # one row per (model, method) for the table.
    primary_rows: list[dict] = []          # one row per model for the metric cards / chart.
    method_eval_rows: list[dict] = []      # for aggregate per-method classification metrics.
    adversarial_rows_by_model: dict[str, list[dict]] = {}
    loaded_models: dict[str, tuple] = {}   # cache for the heatmap section.

    methods_active = (
        {"Threshold", "Consistency", "Logistic"}
        if method_choice == "ALL"
        else {method_choice}
    )
    if "Logistic" in methods_active and logistic_model is None:
        methods_active.discard("Logistic")  # silently skip if no checkpoint.

    for i, m in enumerate(models_to_run):
        status.markdown(
            f"<p style='font-family:\"DM Mono\",monospace; color:#94a3b8;'>→ Loading {m}…</p>",
            unsafe_allow_html=True,
        )

        backbone, processor, device = load_model_by_name(m)
        loaded_models[m] = (backbone, processor)

        captions_to_score = [caption] + generate_adversarial_captions(caption)

        inputs = processor(
            text=captions_to_score,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = backbone(**inputs)

        sims = compute_similarity(outputs.image_embeds, outputs.text_embeds)
        score_values = sims[0].detach().cpu().tolist()
        orig_score, attack_scores = split_similarity_scores(score_values)

        # ---- Per-method decisions for the original caption ----
        per_model_rows: list[dict] = []
        if "Threshold" in methods_active:
            d = method_threshold(orig_score)
            per_model_rows.append({
                "Model": m, "Method": "Threshold",
                "Score": round(orig_score, 4), "Consistency": None, "Decision": d,
            })

        if "Consistency" in methods_active:
            d, c = method_consistency_threshold(orig_score, attack_scores)
            per_model_rows.append({
                "Model": m, "Method": "Consistency",
                "Score": round(orig_score, 4), "Consistency": round(c, 4), "Decision": d,
            })

        if "Logistic" in methods_active:
            d, c = method_logistic(logistic_model, orig_score, attack_scores)
            per_model_rows.append({
                "Model": m, "Method": "Logistic",
                "Score": round(orig_score, 4), "Consistency": round(c, 4), "Decision": d,
            })

        if per_model_rows:
            summary_rows.extend(per_model_rows)
            primary_rows.append(per_model_rows[0])

        # ---- Per-caption labels for aggregate metrics ----
        for j in range(len(captions_to_score)):
            candidate_score = score_values[j]
            comparison_scores = [score_values[k] for k in range(len(score_values)) if k != j]
            y_true = LIKELY_CORRECT if j == 0 else HALLUCINATION

            if "Threshold" in methods_active:
                method_eval_rows.append({
                    "model": m, "method": "Threshold",
                    "y_true": y_true, "y_pred": method_threshold(candidate_score),
                })
            if "Consistency" in methods_active:
                pred, _ = method_consistency_threshold(candidate_score, comparison_scores)
                method_eval_rows.append({
                    "model": m, "method": "Consistency",
                    "y_true": y_true, "y_pred": pred,
                })
            if "Logistic" in methods_active:
                pred, _ = method_logistic(logistic_model, candidate_score, comparison_scores)
                method_eval_rows.append({
                    "model": m, "method": "Logistic",
                    "y_true": y_true, "y_pred": pred,
                })

        # ---- Adversarial breakdown ----
        if attack_mode:
            attack_labels = ["Original", "Swap Attack", "Spaceship Attack", "Dinosaur Attack"]
            rows = []
            for j, cap in enumerate(captions_to_score):
                s = score_values[j]
                rows.append({
                    "Type": attack_labels[j] if j < len(attack_labels) else f"Attack {j}",
                    "Caption": (cap[:70] + "…") if len(cap) > 70 else cap,
                    "Score": round(s, 4),
                    "Decision": method_threshold(s),
                })
            adversarial_rows_by_model[m] = rows

        progress.progress((i + 1) / len(models_to_run))

    status.empty()
    progress.empty()

    # ---- Metric cards ----
    if primary_rows:
        cols = st.columns(len(primary_rows), gap="medium")
        for col, res in zip(cols, primary_rows):
            is_hall = "hallucination" in res["Decision"].lower()
            border = "rgba(239,68,68,0.4)" if is_hall else "rgba(34,197,94,0.4)"
            glow = "rgba(239,68,68,0.35)" if is_hall else "rgba(34,197,94,0.35)"
            score_clr = "#f87171" if is_hall else "#4ade80"
            icon = "⚠️" if is_hall else "✅"

            score_label = "Consistency" if res["Method"] == "Consistency" else "Score"
            score_value = res["Consistency"] if res["Method"] == "Consistency" else res["Score"]

            col.markdown(
                f"""
            <div style="background:rgba(255,255,255,0.04); border:1px solid {border};
                border-radius:20px; padding:1.6rem 1.4rem; box-shadow:0 0 30px {glow};
                text-align:center; animation:fadeSlideUp 0.5s ease both;">
              <p style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;
                letter-spacing:0.12em; margin:0 0 6px; font-weight:700;">
                {res['Model']} · {res['Method']}
              </p>
              <p style="font-size:0.68rem; color:#94a3b8; text-transform:uppercase;
                letter-spacing:0.08em; margin:0 0 4px;">{score_label}</p>
              <p style="font-family:'DM Mono',monospace; font-size:2rem; color:{score_clr};
                margin:0; font-weight:500; line-height:1.1;">{score_value}</p>
              <p style="font-size:0.82rem; color:#e2e8f0; margin:8px 0 0;">{icon} {res['Decision']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ---- Summary table + chart ----
    if summary_rows:
        st.table(pd.DataFrame(summary_rows)[["Model", "Method", "Score", "Consistency", "Decision"]])
        chart_col = "Consistency" if method_choice in {"Consistency", "Logistic"} else "Score"
        chart_df = pd.DataFrame(primary_rows).set_index("Model")
        if chart_col in chart_df.columns and chart_df[chart_col].notna().any():
            st.bar_chart(chart_df[chart_col])

    # ---- Insight panel ----
    #
    # Two modes:
    #   - "single config"  → one model + one method selected. Show the
    #     metrics for that single configuration with a clear interpretation.
    #     No "best vs others" framing — there's nothing to compare.
    #   - "comparison"     → "ALL" picked for either models or methods.
    #     Group metrics per (model, method) combo, highlight the best, and
    #     show every other configuration in a sorted table.
    is_comparing = (model_choice == "ALL") or (method_choice == "ALL")

    metrics_df = pd.DataFrame()
    if method_eval_rows:
        eval_df = pd.DataFrame(method_eval_rows)
        group_keys = ["model", "method"] if is_comparing else ["method"]
        rows = []
        for keys, group in eval_df.groupby(group_keys, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {k: v for k, v in zip(group_keys, keys)}
            if "model" not in row:
                row["model"] = models_to_run[0]
            row.update(compute_classification_metrics(
                group["y_true"].tolist(),
                group["y_pred"].tolist(),
                hallucination_label=HALLUCINATION,
            ))
            rows.append(row)
        metrics_df = pd.DataFrame(rows)

    _section_header("Summary Insight", "#34d399", "#38bdf8")

    def _performance_badge(f1: float) -> None:
        """Coloured badge + plain-English interpretation under the metrics."""
        if f1 >= 0.85:
            st.success(
                "**Strong performance.** The detector reliably distinguishes real "
                "captions from hallucinated ones."
            )
        elif f1 >= 0.65:
            st.warning(
                "**Moderate performance.** The detector catches most hallucinations "
                "but mislabels some captions."
            )
        else:
            st.error(
                "**Weak performance.** Predictions are close to chance on this run."
            )

    if metrics_df.empty:
        st.info("Summary insight is unavailable — no method metrics were produced.")

    elif not is_comparing:
        # ----- Single (model, method) configuration -----
        row = metrics_df.iloc[0]
        f1, precision, recall, accuracy = (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["accuracy"]),
        )
        method_name = str(row["method"])
        model_name = str(row.get("model", models_to_run[0]))

        st.markdown(
            f"Detection performance for the **{method_name}** method on the "
            f"**{model_name}** backbone:"
        )

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        c1.metric("F1",        f"{f1:.2f}")
        c2.metric("Precision", f"{precision:.2f}")
        c3.metric("Recall",    f"{recall:.2f}")
        c4.metric("Accuracy",  f"{accuracy:.2f}")

        _performance_badge(f1)

    else:
        # ----- Multi-config comparison -----
        sorted_df = metrics_df.sort_values("f1", ascending=False).reset_index(drop=True)
        best = sorted_df.iloc[0]
        best_f1 = float(best["f1"])
        best_method = str(best["method"])
        best_model = str(best.get("model", models_to_run[0]))

        # Headline tiles — best config + how many configs we compared.
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.metric("Best Configuration", f"{best_model} · {best_method}")
        c2.metric("Best F1", f"{best_f1:.2f}")
        c3.metric("Configurations Compared", str(len(sorted_df)))

        # Lift over the second-best — gives the user a sense of how decisive the win is.
        if len(sorted_df) > 1:
            second = sorted_df.iloc[1]
            lift = best_f1 - float(second["f1"])
            st.markdown(
                f"**{best_model} · {best_method}** leads with F1 = **{best_f1:.2f}**, "
                f"{lift:+.2f} ahead of the runner-up "
                f"(**{second.get('model', best_model)} · {second['method']}**, "
                f"F1 = {float(second['f1']):.2f})."
            )

        # Full sorted comparison table.
        display_df = sorted_df.copy()
        for col in ("f1", "precision", "recall", "accuracy"):
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        column_order = [c for c in
                        ["model", "method", "f1", "precision", "recall", "accuracy"]
                        if c in display_df.columns]
        display_df = display_df[column_order].rename(
            columns={
                "model": "Model", "method": "Method",
                "f1": "F1", "precision": "Precision",
                "recall": "Recall", "accuracy": "Accuracy",
            }
        )
        st.dataframe(display_df, hide_index=True, width="stretch")

        _performance_badge(best_f1)

    # ---- Adversarial section ----
    if attack_mode and adversarial_rows_by_model:
        _section_header("⚔️ Adversarial Caption Analysis", "#f472b6", "#f59e0b")
        st.markdown(
            "<p style='color:#94a3b8;'>"
            "Original caption vs. three adversarial attacks. A well-calibrated detector "
            "should rank the original highest."
            "</p>",
            unsafe_allow_html=True,
        )
        for model_name, rows in adversarial_rows_by_model.items():
            st.markdown(
                f"<p style='font-weight:700; color:#c4b5fd; margin:1rem 0 0.4rem;'>{model_name}</p>",
                unsafe_allow_html=True,
            )
            st.table(rows)
            adv_df = pd.DataFrame(rows)
            st.bar_chart(adv_df.set_index("Type")["Score"])

    # ---- Heatmaps ----
    if show_heatmaps:
        _section_header("🌡️ Attention Heatmaps", "#f472b6", "#38bdf8")
        st.markdown(
            "<p style='color:#94a3b8;'>"
            "Patch-level activation norms from each model's vision transformer. "
            "Brighter regions = stronger visual signal."
            "</p>",
            unsafe_allow_html=True,
        )

        image_cv = pil_to_cv2(image)
        heatmap_cols = st.columns(len(models_to_run) + 1, gap="medium")

        with heatmap_cols[0]:
            st.image(image, caption="Original Image", width="stretch")

        for col_idx, m in enumerate(models_to_run):
            backbone, processor = loaded_models[m]
            with heatmap_cols[col_idx + 1]:
                with st.spinner(f"Generating {m} heatmap…"):
                    heatmap = generate_heatmap(backbone, processor, image, caption, model_name=m)
                if heatmap is not None:
                    overlay = _render_heatmap_overlay(image_cv, heatmap)
                    st.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption=f"{m} Attention",
                        width="stretch",
                    )
                else:
                    st.warning(f"{m} heatmap failed.")

    # ---- Footer ----
    st.markdown(
        "<div style='margin-top:3rem; height:1px; background:linear-gradient(90deg,"
        "transparent,rgba(124,58,237,0.6),rgba(14,165,233,0.6),transparent);'></div>"
        "<p style='text-align:center; font-family:\"DM Mono\",monospace; font-size:0.72rem;"
        " color:rgba(148,163,184,0.5); margin-top:1rem;'>"
        "VLM Hallucination Detector · CLIP · BLIP · SigLIP</p>",
        unsafe_allow_html=True,
    )

elif run_button:
    st.warning("⚠️ Please provide both an image and a caption before running.")
