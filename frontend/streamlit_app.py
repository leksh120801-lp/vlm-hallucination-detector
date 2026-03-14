import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

from models.clip_model import load_clip_model
from utils.similarity import compute_similarity, detect_hallucination
from utils.preprocessing import pil_to_cv2
from utils.visualization import generate_fake_heatmap
from utils.caption_attack import generate_adversarial_captions
from models.model_registry import load_model_by_name


# -----------------------------
# Streamlit UI Title
# -----------------------------

st.title("VLM Hallucination Detector")

st.write(
    "Upload an image and caption to test whether the caption is hallucinated "
    "by a vision-language model."
)


# -----------------------------
# Load model (cached)
# -----------------------------
model_choice = st.selectbox(
    "Select Model",
    ["CLIP","BLIP"]
)

from models.model_registry import load_model_by_name

@st.cache_resource
def load_model(name):
    return load_model_by_name(name)

model, processor, device = load_model(model_choice)


# -----------------------------
# Model selection (future use)
# -----------------------------

model_choice = st.selectbox(
    "Select Model",
    ["CLIP"]
)


# -----------------------------
# Upload Image
# -----------------------------

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)


# -----------------------------
# Caption Input
# -----------------------------

caption = st.text_input("Enter caption")


# -----------------------------
# Adversarial Attack Toggle
# -----------------------------

attack_mode = st.checkbox("Run adversarial caption test")


# -----------------------------
# Run Button
# -----------------------------

run_button = st.button("Run Hallucination Detection")


# -----------------------------
# Inference
# -----------------------------

if run_button and uploaded_image and caption:

    image = Image.open(uploaded_image).convert("RGB")

    # Generate captions
    if attack_mode:
        attacks = generate_adversarial_captions(caption)
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

    similarity = compute_similarity(
        outputs.image_embeds,
        outputs.text_embeds
    )

    # -----------------------------
    # Results Table
    # -----------------------------

    results = []

    for i, test_caption in enumerate(captions):

        score = similarity[0][i].item()

        decision = detect_hallucination(score)

        results.append({
            "caption": test_caption,
            "score": round(score, 3),
            "decision": decision
        })

    st.subheader("Results")

    st.table(results)

    # -----------------------------
    # Display main metrics
    # -----------------------------

    main_score = similarity[0][0].item()
    main_decision = detect_hallucination(main_score)

    st.metric("Similarity Score", round(main_score, 3))
    st.metric("Prediction", main_decision)

    # -----------------------------
    # Heatmap Visualization
    # -----------------------------

    st.subheader("Attention Heatmap")

    image_cv = pil_to_cv2(image)

    heatmap = generate_fake_heatmap()

    heatmap = cv2.resize(
        heatmap,
        (image_cv.shape[1], image_cv.shape[0])
    )

    overlay = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image_cv,
        0.6,
        overlay,
        0.4,
        0
    )

    st.image(overlay, caption="Model Attention Heatmap")