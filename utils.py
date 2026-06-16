"""
utils.py
Fungsi-fungsi pembantu: konversi gambar, plotting histogram & spektrum FFT.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def pil_to_cv2(pil_image):
    """Konversi PIL Image ke format OpenCV (BGR numpy array)."""
    img_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Konversi citra OpenCV (BGR) ke PIL Image (RGB)."""
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def resize_if_large(image, max_dim=1000):
    """Resize gambar jika dimensi terbesar melebihi max_dim (efisiensi proses)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def plot_histogram(image, title="Histogram"):
    """Membuat figure matplotlib untuk histogram citra."""
    fig, ax = plt.subplots(figsize=(5, 3))

    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color="black")
    else:
        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)

    ax.set_title(title)
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")
    ax.set_xlim([0, 256])
    fig.tight_layout()
    return fig


def plot_fft_spectrum(magnitude_spectrum, title="Spektrum FFT"):
    """Membuat figure matplotlib untuk visualisasi spektrum FFT."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(magnitude_spectrum, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def image_to_bytes(cv2_image, ext=".png"):
    """Konversi citra OpenCV ke bytes untuk fitur download."""
    is_success, buffer = cv2.imencode(ext, cv2_image)
    return io.BytesIO(buffer).getvalue() if is_success else None
