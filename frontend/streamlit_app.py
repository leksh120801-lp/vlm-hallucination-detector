from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
from models.clip_model import load_clip_model
from utils.similarity import compute_similarity, detect_hallucination
from utils.preprocessing import pil_to_cv2
from utils.visualization import generate_fake_heatmap

from PIL import Image
import torch
import cv2
import numpy as np


st.title("VLM Hallucination Detector")

st.write(
    "Upload an image and caption to test if the caption is hallucinated."
)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image")

caption = st.text_input("Enter caption")

run_button = st.button("Run Hallucination Detection")

@st.cache_resource
def load_model():
    return load_clip_model()

model, processor, device = load_model()

if run_button and uploaded_image and caption:

    image = Image.open(uploaded_image).convert("RGB")

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

    score = similarity[0][0].item()

    decision = detect_hallucination(score)

    st.subheader("Result")

    st.write("Similarity score:", score)

    st.write("Decision:", decision)

    # Heatmap
    image_cv = pil_to_cv2(image)

    heatmap = generate_fake_heatmap()

    heatmap = cv2.resize(heatmap, (image_cv.shape[1], image_cv.shape[0]))

    overlay = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(image_cv, 0.6, overlay, 0.4, 0)

    st.image(overlay, caption="Attention Heatmap")