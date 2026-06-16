"""
image_processor.py
Modul inti pemrosesan citra untuk Aplikasi Filter Foto Sederhana.
Final Project - Pengolahan Citra Digital
"""

import cv2
import numpy as np


# =========================================================
# 1. LOW-PASS FILTER (BLUR / SMOOTHING)
# =========================================================

def gaussian_blur(image, kernel_size=5, sigma=0):
    """Menerapkan Gaussian Blur (low-pass filter)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def average_filter(image, kernel_size=5):
    """Menerapkan Average (Mean) Filter."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.blur(image, (kernel_size, kernel_size))


def median_filter(image, kernel_size=5):
    """Menerapkan Median Filter, efektif untuk salt & pepper noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter - smoothing yang mempertahankan tepi (edge-preserving)."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# =========================================================
# 2. HIGH-PASS FILTER (SHARPENING)
# =========================================================

def unsharp_masking(image, kernel_size=9, sigma=10, amount=1.0):
    """
    Unsharp Masking: citra_asli + amount * (citra_asli - citra_blur)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def laplacian_sharpening(image, strength=1.0):
    """Sharpening menggunakan operator Laplacian."""
    gray_mode = len(image.shape) == 2

    if gray_mode:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image.astype(np.float64) - strength * laplacian
    else:
        sharpened = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            laplacian = cv2.Laplacian(channel, cv2.CV_64F)
            sharpened[:, :, c] = channel.astype(np.float64) - strength * laplacian

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


# =========================================================
# 3. FFT (FAST FOURIER TRANSFORM)
# =========================================================

def compute_fft_spectrum(image):
    """
    Menghitung magnitude spectrum FFT dari sebuah citra grayscale.
    Mengembalikan array float (belum dinormalisasi ke 0-255) untuk plotting.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum


def apply_frequency_filter(image, filter_type="lowpass", radius=30):
    """
    Menerapkan filter pada domain frekuensi (FFT) secara langsung.
    filter_type: 'lowpass' atau 'highpass'
    radius: radius mask lingkaran di tengah spektrum
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    if filter_type == "lowpass":
        fshift_filtered = fshift * mask
    elif filter_type == "highpass":
        fshift_filtered = fshift * (1 - mask)
    else:
        raise ValueError("filter_type harus 'lowpass' atau 'highpass'")

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)


# =========================================================
# 4. KONTRAS & BRIGHTNESS
# =========================================================

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Penyesuaian kontras (alpha) dan brightness (beta) secara manual.
    new_pixel = alpha * pixel + beta
    alpha > 1 -> kontras meningkat, alpha < 1 -> kontras menurun
    beta > 0  -> lebih terang, beta < 0 -> lebih gelap
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def histogram_equalization(image):
    """
    Histogram Equalization untuk meningkatkan kontras secara otomatis.
    Bekerja pada channel luminance (Y) jika citra berwarna.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def clahe_equalization(image, clip_limit=2.0, tile_grid_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Versi adaptif dari histogram equalization yang bekerja per-region.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid_size, tile_grid_size))

    if len(image.shape) == 2:
        return clahe.apply(image)

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# =========================================================
# 5. FUNGSI UTILITAS
# =========================================================

def get_histogram(image):
    """
    Mengembalikan histogram per-channel.
    Untuk citra berwarna: dict {'B': hist, 'G': hist, 'R': hist}
    Untuk citra grayscale: dict {'Gray': hist}
    """
    histograms = {}
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        histograms["Gray"] = hist.flatten()
    else:
        channels = {"B": 0, "G": 1, "R": 2}
        for name, idx in channels.items():
            hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
            histograms[name] = hist.flatten()
    return histograms


# =========================================================
# 6. DISPATCHER (mempermudah pemanggilan dari UI)
# =========================================================

FILTER_FUNCTIONS = {
    "Gaussian Blur": gaussian_blur,
    "Average Filter": average_filter,
    "Median Filter": median_filter,
    "Bilateral Filter": bilateral_filter,
    "Unsharp Masking": unsharp_masking,
    "Laplacian Sharpening": laplacian_sharpening,
    "Histogram Equalization": lambda img, **kw: histogram_equalization(img),
    "CLAHE": lambda img, **kw: clahe_equalization(img, kw.get("clip_limit", 2.0)),
    "Brightness & Contrast": lambda img, **kw: adjust_brightness_contrast(
        img, kw.get("alpha", 1.0), kw.get("beta", 0)
    ),
}


def apply_filter(image, filter_name, **kwargs):
    """Dispatcher utama: menerapkan filter berdasarkan nama."""
    func = FILTER_FUNCTIONS.get(filter_name)
    if func is None:
        raise ValueError(f"Filter '{filter_name}' tidak dikenali.")
    return func(image, **kwargs)
