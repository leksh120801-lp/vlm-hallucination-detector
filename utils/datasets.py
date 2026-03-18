from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO


# -----------------------------
# SAFE FLICKR (manual URLs fallback)
# -----------------------------

def load_flickr_sample(n=10):

    dataset = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        trust_remote_code=True
    )

    samples = []

    for item in dataset[:n]:

        url = item["url"]
        caption = item["sentences"][0]

        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            samples.append({
                "image": image,
                "caption": caption
            })

        except:
            continue

    return samples


# -----------------------------
# SAFE COCO (Karpathy split)
# -----------------------------

def load_coco_sample(n=10):

    dataset = load_dataset(
        "yerevann/coco-karpathy",
        split="validation[:50]"
    )

    samples = []

    for item in dataset:

        url = item["url"]
        caption = item["sentences"][0]

        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            samples.append({
                "image": image,
                "caption": caption
            })

        except:
            continue

        if len(samples) >= n:
            break

    return samples


# -----------------------------
# REPLACE NOCAPS (UNSTABLE) WITH VISUAL GENOME SAMPLE
# -----------------------------

def load_visual_genome_sample(n=10):

    dataset = load_dataset(
        "visual_genome",
        "region_descriptions_v1.2.0",
        split="train[:50]"
    )

    samples = []

    for item in dataset:

        url = item["image"]["url"]
        caption = item["regions"][0]["phrase"]

        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            samples.append({
                "image": image,
                "caption": caption
            })

        except:
            continue

        if len(samples) >= n:
            break

    return samples