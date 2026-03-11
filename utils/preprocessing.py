from PIL import Image
import cv2
import numpy as np

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

#This loads images safely.

import requests
from io import BytesIO


def load_image_from_url(url):

    response = requests.get(url)

    image = Image.open(BytesIO(response.content)).convert("RGB")

    return image



def pil_to_cv2(image):

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)