"""
app.py
Aplikasi Filter Foto Sederhana - Final Project Pengolahan Citra Digital
Kelas I243H - Semester Genap 2025/2026

Cara menjalankan:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2

from image_processor import (
    gaussian_blur,
    average_filter,
    median_filter,
    bilateral_filter,
    unsharp_masking,
    laplacian_sharpening,
    adjust_brightness_contrast,
    histogram_equalization,
    clahe_equalization,
    compute_fft_spectrum,
    apply_frequency_filter,
)
from utils import (
    pil_to_cv2,
    cv2_to_pil,
    resize_if_large,
    plot_histogram,
    plot_fft_spectrum,
    image_to_bytes,
)
from PIL import Image


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Aplikasi Filter Foto Sederhana",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Aplikasi Filter Foto Sederhana")
st.caption("Final Project - Pengolahan Citra Digital | Kelas I243H | 2025/2026")
st.markdown(
    "Aplikasi untuk menerapkan filter **blur**, **sharpen**, **smoothing**, "
    "serta pengaturan **kontras & brightness** pada gambar, "
    "dilengkapi analisis **histogram** dan **FFT**."
)
st.divider()


# =========================================================
# SIDEBAR - UPLOAD & PILIHAN FILTER
# =========================================================
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload Gambar", type=["jpg", "jpeg", "png"]
)

filter_category = st.sidebar.selectbox(
    "Kategori Filter",
    [
        "Low-pass Filter (Blur/Smoothing)",
        "High-pass Filter (Sharpening)",
        "Kontras & Brightness",
        "Filter Domain Frekuensi (FFT)",
    ],
)

# --- Parameter dinamis berdasarkan kategori ---
filter_name = None
params = {}

if filter_category == "Low-pass Filter (Blur/Smoothing)":
    filter_name = st.sidebar.radio(
        "Pilih Filter", ["Gaussian Blur", "Average Filter", "Median Filter", "Bilateral Filter"]
    )
    if filter_name in ["Gaussian Blur", "Average Filter", "Median Filter"]:
        params["kernel_size"] = st.sidebar.slider("Kernel Size", 1, 31, 5, step=2)
        if filter_name == "Gaussian Blur":
            params["sigma"] = st.sidebar.slider("Sigma", 0, 20, 0)
    elif filter_name == "Bilateral Filter":
        params["d"] = st.sidebar.slider("Diameter (d)", 1, 25, 9)
        params["sigma_color"] = st.sidebar.slider("Sigma Color", 1, 150, 75)
        params["sigma_space"] = st.sidebar.slider("Sigma Space", 1, 150, 75)

elif filter_category == "High-pass Filter (Sharpening)":
    filter_name = st.sidebar.radio(
        "Pilih Filter", ["Unsharp Masking", "Laplacian Sharpening"]
    )
    if filter_name == "Unsharp Masking":
        params["kernel_size"] = st.sidebar.slider("Kernel Size", 1, 31, 9, step=2)
        params["sigma"] = st.sidebar.slider("Sigma Blur", 1, 30, 10)
        params["amount"] = st.sidebar.slider("Amount (kekuatan)", 0.1, 3.0, 1.0, step=0.1)
    elif filter_name == "Laplacian Sharpening":
        params["strength"] = st.sidebar.slider("Strength", 0.1, 3.0, 1.0, step=0.1)

elif filter_category == "Kontras & Brightness":
    filter_name = st.sidebar.radio(
        "Pilih Metode", ["Manual (Alpha & Beta)", "Histogram Equalization", "CLAHE"]
    )
    if filter_name == "Manual (Alpha & Beta)":
        params["alpha"] = st.sidebar.slider("Kontras (alpha)", 0.1, 3.0, 1.0, step=0.1)
        params["beta"] = st.sidebar.slider("Brightness (beta)", -100, 100, 0)
    elif filter_name == "CLAHE":
        params["clip_limit"] = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0, step=0.5)

elif filter_category == "Filter Domain Frekuensi (FFT)":
    filter_name = st.sidebar.radio("Pilih Filter Frekuensi", ["Low-pass (FFT)", "High-pass (FFT)"])
    params["radius"] = st.sidebar.slider("Radius Mask", 5, 200, 30)


# =========================================================
# PROSES UTAMA
# =========================================================
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    original = pil_to_cv2(pil_image)
    original = resize_if_large(original, max_dim=1000)

    # Terapkan filter sesuai pilihan
    if filter_category == "Low-pass Filter (Blur/Smoothing)":
        if filter_name == "Gaussian Blur":
            result = gaussian_blur(original, params["kernel_size"], params["sigma"])
        elif filter_name == "Average Filter":
            result = average_filter(original, params["kernel_size"])
        elif filter_name == "Median Filter":
            result = median_filter(original, params["kernel_size"])
        else:
            result = bilateral_filter(original, params["d"], params["sigma_color"], params["sigma_space"])

    elif filter_category == "High-pass Filter (Sharpening)":
        if filter_name == "Unsharp Masking":
            result = unsharp_masking(original, params["kernel_size"], params["sigma"], params["amount"])
        else:
            result = laplacian_sharpening(original, params["strength"])

    elif filter_category == "Kontras & Brightness":
        if filter_name == "Manual (Alpha & Beta)":
            result = adjust_brightness_contrast(original, params["alpha"], params["beta"])
        elif filter_name == "Histogram Equalization":
            result = histogram_equalization(original)
        else:
            result = clahe_equalization(original, params["clip_limit"])

    elif filter_category == "Filter Domain Frekuensi (FFT)":
        ftype = "lowpass" if filter_name == "Low-pass (FFT)" else "highpass"
        result_gray = apply_frequency_filter(original, ftype, params["radius"])
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)

    # =========================================================
    # TAMPILAN HASIL
    # =========================================================
    st.subheader("📷 Perbandingan Gambar")
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2_to_pil(original), caption="Gambar Asli", use_container_width=True)

    with col2:
        st.image(cv2_to_pil(result), caption=f"Hasil: {filter_name}", use_container_width=True)

    # Tombol download
    result_bytes = image_to_bytes(result, ext=".png")
    st.download_button(
        label="⬇️ Download Hasil",
        data=result_bytes,
        file_name=f"hasil_{filter_name.replace(' ', '_').lower()}.png",
        mime="image/png",
    )

    st.divider()

    # =========================================================
    # ANALISIS: HISTOGRAM
    # =========================================================
    st.subheader("📊 Analisis Histogram")
    col3, col4 = st.columns(2)

    with col3:
        fig_hist_original = plot_histogram(original, "Histogram - Gambar Asli")
        st.pyplot(fig_hist_original)

    with col4:
        fig_hist_result = plot_histogram(result, "Histogram - Hasil Filter")
        st.pyplot(fig_hist_result)

    st.divider()

    # =========================================================
    # ANALISIS: SPEKTRUM FFT
    # =========================================================
    st.subheader("🌊 Analisis Spektrum Frekuensi (FFT)")
    col5, col6 = st.columns(2)

    with col5:
        spectrum_original = compute_fft_spectrum(original)
        fig_fft_original = plot_fft_spectrum(spectrum_original, "Spektrum FFT - Gambar Asli")
        st.pyplot(fig_fft_original)

    with col6:
        spectrum_result = compute_fft_spectrum(result)
        fig_fft_result = plot_fft_spectrum(spectrum_result, "Spektrum FFT - Hasil Filter")
        st.pyplot(fig_fft_result)

    st.info(
        "💡 **Catatan analisis:** Pada filter low-pass, energi frekuensi tinggi "
        "(area tepi spektrum) tampak lebih redup dibanding gambar asli. "
        "Sebaliknya, pada filter high-pass / sharpening, area tepi spektrum "
        "tampak lebih terang karena komponen frekuensi tinggi diperkuat."
    )

else:
    st.info("👈 Silakan upload gambar pada sidebar untuk mulai memproses.")
    st.markdown(
        """
        ### Fitur Aplikasi
        - **Low-pass Filter**: Gaussian Blur, Average Filter, Median Filter, Bilateral Filter
        - **High-pass Filter**: Unsharp Masking, Laplacian Sharpening
        - **Kontras & Brightness**: Pengaturan manual, Histogram Equalization, CLAHE
        - **Domain Frekuensi**: Filter Low-pass/High-pass berbasis FFT
        - **Analisis**: Histogram dan Spektrum FFT sebelum/sesudah filter
        - **Download** hasil gambar yang telah diproses
        """
    )
