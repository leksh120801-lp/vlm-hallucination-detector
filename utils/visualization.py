import matplotlib.pyplot as plt
import numpy as np


def show_attention_heatmap(image, similarity_score):

    plt.imshow(image)
    plt.title(f"Similarity Score: {similarity_score:.2f}")
    plt.axis("off")

    plt.show()