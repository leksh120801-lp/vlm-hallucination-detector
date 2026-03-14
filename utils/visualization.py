import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_heatmap(image, heatmap):

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize and convert to color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend image and heatmap
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay


def show_heatmap(image, heatmap):

    overlay = overlay_heatmap(image, heatmap)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def save_heatmap(image, heatmap, filename):

    overlay = overlay_heatmap(image, heatmap)

    cv2.imwrite(filename, overlay)