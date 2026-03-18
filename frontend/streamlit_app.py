import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from models.model_registry import load_model_by_name
from utils.similarity import compute_similarity, detect_hallucination
from utils.preprocessing import pil_to_cv2
from utils.caption_attack import generate_adversarial_captions
from utils.real_heatmap import generate_clip_heatmap
from utils.datasets import load_visual_genome_sample

from utils.datasets import (
    load_flickr_sample,
    load_coco_sample,
    load_visual_genome_sample
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="VLM Hallucination Detector",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────
# GLASSMORPHISM CSS + ANIMATIONS
# ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── ANIMATED BACKGROUND ── */
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

/* Floating orbs via pseudo-element on main block */
[data-testid="stMain"]::before {
    content: "";
    position: fixed;
    top: -150px; left: -150px;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%);
    animation: orb1 14s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"]::after {
    content: "";
    position: fixed;
    bottom: -100px; right: -100px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,0.18) 0%, transparent 70%);
    animation: orb2 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes orb1 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(80px, 60px) scale(1.2); }
}

@keyframes orb2 {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-60px, -80px) scale(1.15); }
}

/* ── HIDE DEFAULT STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── GLOBAL FONT ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    color: #e2e8f0;
}

/* ── GLASS CARD MIXIN (applied to main blocks) ── */
[data-testid="stVerticalBlock"] > [data-testid="element-container"],
[data-testid="stForm"],
section[data-testid="stSidebar"] {
    backdrop-filter: blur(16px) saturate(160%);
    -webkit-backdrop-filter: blur(16px) saturate(160%);
}

/* ── TITLE ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 3rem !important;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    animation: titleShimmer 6s linear infinite;
    background-size: 200%;
}

@keyframes titleShimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #c4b5fd, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── GLASS CONTAINERS ── */
div[data-testid="stExpander"],
div.stForm,
div[data-testid="metric-container"],
div.element-container {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    transition: box-shadow 0.35s ease, border-color 0.35s ease;
}

div[data-testid="stExpander"]:hover,
div.element-container:hover {
    border-color: rgba(167, 139, 250, 0.35) !important;
    box-shadow: 0 0 28px rgba(124, 58, 237, 0.18), 0 0 6px rgba(167, 139, 250, 0.15);
}

/* ── SELECTBOX / RADIO ── */
div[data-baseweb="select"] > div,
div[data-baseweb="radio"] {
    background: rgba(15, 10, 40, 0.6) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(167, 139, 250, 0.7) !important;
    box-shadow: 0 0 18px rgba(124, 58, 237, 0.25) !important;
}

/* ── TEXT INPUT ── */
input[type="text"], textarea {
    background: rgba(15, 10, 40, 0.55) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

input[type="text"]:focus, textarea:focus {
    border-color: rgba(56, 189, 248, 0.7) !important;
    box-shadow: 0 0 20px rgba(0, 200, 255, 0.2) !important;
    outline: none !important;
}

/* ── SLIDER ── */
div[data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(135deg, #7c3aed, #0ea5e9) !important;
    box-shadow: 0 0 14px rgba(124, 58, 237, 0.7) !important;
    border: 2px solid rgba(255,255,255,0.25) !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, rgba(124,58,237,0.5), rgba(14,165,233,0.5)) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease, background 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    background: rgba(124,58,237,0.07) !important;
    border-color: rgba(167, 139, 250, 0.8) !important;
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.2);
}

/* ── RUN BUTTON ── */
div.stButton > button {
    background: linear-gradient(135deg, #6d28d9 0%, #0369a1 50%, #6d28d9 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: background-position 0.6s ease, box-shadow 0.4s ease, transform 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(109, 40, 217, 0.45), 0 0 0 1px rgba(255,255,255,0.08) !important;
}

div.stButton > button::before {
    content: "";
    position: absolute;
    top: -50%; left: -75%;
    width: 50%; height: 200%;
    background: rgba(255,255,255,0.12);
    transform: skewX(-20deg);
    transition: left 0.5s ease;
}

div.stButton > button:hover {
    background-position: right center !important;
    box-shadow: 0 6px 36px rgba(109, 40, 217, 0.65), 0 0 0 1px rgba(255,255,255,0.15) !important;
    transform: translateY(-2px) !important;
}

div.stButton > button:hover::before {
    left: 125%;
}

div.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
}

/* ── CHECKBOX ── */
label[data-baseweb="checkbox"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    color: #c4b5fd !important;
}

input[type="checkbox"]:checked + div {
    background: linear-gradient(135deg, #7c3aed, #0ea5e9) !important;
    border-color: transparent !important;
}

/* ── TABLE ── */
[data-testid="stTable"] table {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    width: 100% !important;
}

[data-testid="stTable"] thead tr th {
    background: rgba(124,58,237,0.25) !important;
    color: #c4b5fd !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 12px 20px !important;
    border-bottom: 1px solid rgba(167,139,250,0.3) !important;
}

[data-testid="stTable"] tbody tr {
    transition: background 0.25s ease;
    animation: rowFadeIn 0.5s ease both;
}

[data-testid="stTable"] tbody tr:nth-child(1)  { animation-delay: 0.05s; }
[data-testid="stTable"] tbody tr:nth-child(2)  { animation-delay: 0.15s; }
[data-testid="stTable"] tbody tr:nth-child(3)  { animation-delay: 0.25s; }

@keyframes rowFadeIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

[data-testid="stTable"] tbody tr:hover {
    background: rgba(124,58,237,0.12) !important;
}

[data-testid="stTable"] tbody tr td {
    padding: 11px 20px !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    color: #e2e8f0 !important;
}

/* ── BAR CHART ── */
[data-testid="stVegaLiteChart"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(167,139,250,0.15) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    animation: fadeSlideUp 0.7s ease both;
    animation-delay: 0.3s;
}

/* ── IMAGES ── */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(167,139,250,0.1) !important;
    animation: fadeSlideUp 0.6s ease both;
}

/* ── INFO / WARNING BOXES ── */
[data-testid="stAlert"] {
    background: rgba(14,165,233,0.08) !important;
    border: 1px solid rgba(14,165,233,0.3) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── SUBHEADER BAR ── */
[data-testid="stSubheader"] h3::after {
    content: "";
    display: block;
    margin-top: 6px;
    height: 2px;
    width: 60px;
    background: linear-gradient(90deg, #7c3aed, #38bdf8, transparent);
    border-radius: 2px;
}

/* ── GLOBAL FADE-IN ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stVerticalBlock"] {
    animation: fadeSlideUp 0.55s ease both;
}

/* ── RADIO BUTTONS ── */
[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    transition: background 0.25s, border-color 0.25s, box-shadow 0.25s !important;
    margin: 4px 0 !important;
    cursor: pointer !important;
}

[data-testid="stRadio"] label:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 16px rgba(124,58,237,0.2) !important;
}

/* ── PROGRESS / SPINNER ── */
[data-testid="stSpinner"] > div {
    border-color: rgba(124,58,237,0.8) transparent transparent transparent !important;
}

/* ── WRITE TEXT ── */
[data-testid="stMarkdownContainer"] p {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# ANIMATED HEADER BANNER
# ─────────────────────────────────────────

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(109,40,217,0.18) 0%, rgba(14,165,233,0.12) 100%);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 24px;
    padding: 2.5rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.7s ease both;
">
  <!-- Decorative glow orb inside header -->
  <div style="
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
    pointer-events: none;
  "></div>

  <div style="display:flex; align-items:center; gap:16px; margin-bottom:0.6rem;">
    <span style="font-size:2.4rem; filter: drop-shadow(0 0 12px rgba(167,139,250,0.8));">🔬</span>
    <h1 style="
      margin:0; padding:0;
      font-family:'Syne',sans-serif; font-weight:800; font-size:2.2rem;
      background: linear-gradient(135deg,#a78bfa 0%,#38bdf8 55%,#f472b6 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text;
    ">VLM Hallucination Detector</h1>
  </div>

  <p style="
    margin:0; font-family:'DM Mono',monospace; font-size:0.88rem;
    color:#94a3b8; letter-spacing:0.02em;
  ">Evaluate vision–language model captions across CLIP · BLIP · SigLIP</p>

  <!-- Thin gradient rule -->
  <div style="
    margin-top:1.4rem; height:1px;
    background: linear-gradient(90deg, rgba(124,58,237,0.6), rgba(14,165,233,0.6), transparent);
  "></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# INPUT MODE
# ─────────────────────────────────────────

_col_mode, _col_model = st.columns([1, 1], gap="large")

with _col_mode:
    st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#7c3aed; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; font-weight:700;">Input Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Input Mode",
        ["Upload Image", "Dataset Sample"],
        label_visibility="collapsed"
    )

with _col_model:
    st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#0ea5e9; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; font-weight:700;">Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model",
        ["CLIP", "BLIP", "SigLIP", "ALL"],
        label_visibility="collapsed"
    )


# ─────────────────────────────────────────
# IMAGE + CAPTION INPUT
# ─────────────────────────────────────────

image = None
caption = None

st.markdown("---")

if mode == "Upload Image":

    col_up, col_cap = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Image</p>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, use_container_width=True)

    with col_cap:
        st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Caption</p>', unsafe_allow_html=True)
        caption = st.text_input("Enter caption", label_visibility="collapsed", placeholder="Describe what you see in the image…")

else:

    col_ds, col_slide = st.columns([1, 2], gap="large")

    with col_ds:
        st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Dataset</p>', unsafe_allow_html=True)
        dataset_choice = st.selectbox(
            "Dataset",
            ["Flickr30k", "COCO", "NoCaps"],
            label_visibility="collapsed"
        )

    if dataset_choice == "Flickr30k":
        samples = load_flickr_sample(10)
    elif dataset_choice == "COCO":
        samples = load_coco_sample(10)
    elif dataset_choice == "NoCaps":
        samples = load_visual_genome_sample(10)

    with col_slide:
        st.markdown('<p style="font-family:\'Syne\',sans-serif; font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">Sample Index</p>', unsafe_allow_html=True)
        idx = st.slider("Select sample", 0, len(samples) - 1, label_visibility="collapsed")

    image   = samples[idx]["image"]
    caption = samples[idx]["caption"]

    col_img, col_cap = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, use_container_width=True)

    with col_cap:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            backdrop-filter: blur(12px);
        ">
          <p style="
            font-family:'Syne',sans-serif; font-size:0.7rem;
            color:#7c3aed; text-transform:uppercase; letter-spacing:0.12em;
            margin:0 0 8px; font-weight:700;
          ">Caption</p>
          <p style="
            font-family:'DM Mono',monospace; font-size:0.9rem;
            color:#e2e8f0; line-height:1.7; margin:0;
          ">{caption}</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# OPTIONS + RUN
# ─────────────────────────────────────────

st.markdown("---")

col_opt, col_btn = st.columns([2, 1], gap="large")

with col_opt:
    attack_mode = st.checkbox("⚔️ Run adversarial caption test", value=False)

with col_btn:
    run_button = st.button("⚡ Run Evaluation", use_container_width=True)


# ─────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────

if run_button and image and caption:

    # ── section header ──
    st.markdown("""
    <div style="
        margin: 2rem 0 1rem;
        display:flex; align-items:center; gap:12px;
    ">
      <div style="
        width:4px; height:28px; border-radius:2px;
        background: linear-gradient(180deg,#7c3aed,#38bdf8);
      "></div>
      <h2 style="
        margin:0; font-family:'Syne',sans-serif; font-weight:700; font-size:1.25rem;
        background: linear-gradient(90deg,#c4b5fd,#7dd3fc);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
      ">Model Comparison</h2>
    </div>
    """, unsafe_allow_html=True)

    models_to_run = ["CLIP", "BLIP", "SigLIP"] if model_choice == "ALL" else [model_choice]
    all_results = []

    progress = st.progress(0)
    status   = st.empty()

    for i, m in enumerate(models_to_run):
        status.markdown(f'<p style="font-family:\'DM Mono\',monospace; color:#94a3b8; font-size:0.82rem;">→ Loading <span style="color:#c4b5fd">{m}</span>…</p>', unsafe_allow_html=True)

        model, processor, device = load_model_by_name(m)

        if attack_mode:
            attacks  = generate_adversarial_captions(caption)
            captions = [caption] + attacks
        else:
            captions = [caption]

        inputs = processor(
            text=captions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        similarity = compute_similarity(outputs.image_embeds, outputs.text_embeds)
        score      = similarity[0][0].item()
        decision   = detect_hallucination(score)

        all_results.append({
            "Model":    m,
            "Score":    round(score, 4),
            "Decision": decision
        })

        progress.progress((i + 1) / len(models_to_run))

    status.empty()
    progress.empty()

    # ── METRIC CARDS ──
    metric_cols = st.columns(len(all_results), gap="medium")
    for col, res in zip(metric_cols, all_results):
        is_hallucination = "Hallucination" in res["Decision"]
        glow_color = "rgba(239,68,68,0.35)" if is_hallucination else "rgba(34,197,94,0.35)"
        border_color = "rgba(239,68,68,0.4)" if is_hallucination else "rgba(34,197,94,0.4)"
        icon = "⚠️" if is_hallucination else "✅"
        score_color = "#f87171" if is_hallucination else "#4ade80"

        col.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid {border_color};
            border-radius: 20px;
            padding: 1.6rem 1.4rem;
            backdrop-filter: blur(14px);
            box-shadow: 0 0 30px {glow_color};
            text-align: center;
            animation: fadeSlideUp 0.5s ease both;
        ">
          <p style="
            font-family:'Syne',sans-serif; font-size:0.72rem;
            color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em;
            margin:0 0 6px; font-weight:700;
          ">{res['Model']}</p>
          <p style="
            font-family:'DM Mono',monospace; font-size:2rem;
            color:{score_color}; margin:0; font-weight:500; line-height:1.1;
          ">{res['Score']}</p>
          <p style="
            font-family:'Syne',sans-serif; font-size:0.82rem;
            color:#e2e8f0; margin:8px 0 0;
          ">{icon} {res['Decision']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── TABLE ──
    st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
    st.table(all_results)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── BAR CHART ──
    import pandas as pd
    df = pd.DataFrame(all_results)
    st.bar_chart(df.set_index("Model")["Score"])

    # ── HEATMAP ──
    if model_choice in ["CLIP", "ALL"]:

        st.markdown("""
        <div style="
            margin: 2rem 0 1rem;
            display:flex; align-items:center; gap:12px;
        ">
          <div style="
            width:4px; height:28px; border-radius:2px;
            background: linear-gradient(180deg,#f472b6,#38bdf8);
          "></div>
          <h2 style="
            margin:0; font-family:'Syne',sans-serif; font-weight:700; font-size:1.25rem;
            background: linear-gradient(90deg,#f9a8d4,#7dd3fc);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
          ">Attention Heatmap — CLIP</h2>
        </div>
        """, unsafe_allow_html=True)

        try:
            clip_model, clip_processor, _ = load_model_by_name("CLIP")

            heatmap  = generate_clip_heatmap(clip_model, clip_processor, image, caption)
            image_cv = pil_to_cv2(image)

            heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))
            heatmap = heatmap / heatmap.max()
            heatmap = heatmap ** 3
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(image_cv, 0.5, heatmap, 0.7, 0)

            col_heat, col_orig = st.columns(2, gap="large")
            with col_heat:
                st.image(overlay, caption="CLIP Attention Overlay", use_container_width=True)
            with col_orig:
                st.image(image,   caption="Original Image",         use_container_width=True)

        except Exception as e:
            st.warning(f"Heatmap failed: {e}")

    else:
        st.info("Heatmap currently supported only for CLIP")

    # ── FOOTER PULSE ──
    st.markdown("""
    <div style="
        margin-top: 3rem;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.6), rgba(14,165,233,0.6), transparent);
        animation: fadeSlideUp 1s ease both;
    "></div>
    <p style="
        text-align:center; font-family:'DM Mono',monospace;
        font-size:0.72rem; color:rgba(148,163,184,0.5);
        margin-top:1rem;
    ">VLM Hallucination Detector · Powered by CLIP · BLIP · SigLIP</p>
    """, unsafe_allow_html=True)