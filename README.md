# 🖼️ Aplikasi Filter Foto Sederhana

Final Project - Pengolahan Citra Digital
Kelas I243H | Semester Genap 2025/2026

## Deskripsi

Aplikasi web sederhana berbasis **Streamlit** yang memberikan efek filter
pada gambar seperti blur, sharpen, dan smoothing, dilengkapi pengaturan
kontras & brightness serta analisis histogram dan spektrum FFT.

## Fitur

- **Low-pass Filter**: Gaussian Blur, Average Filter, Median Filter, Bilateral Filter
- **High-pass Filter**: Unsharp Masking, Laplacian Sharpening
- **Kontras & Brightness**: Manual (alpha/beta), Histogram Equalization, CLAHE
- **Filter Domain Frekuensi**: Low-pass & High-pass berbasis FFT
- Visualisasi histogram dan spektrum FFT (sebelum & sesudah filter)
- Download hasil gambar yang telah diproses

## Struktur Project

```
filter_foto/
├── app.py                # Antarmuka Streamlit (entry point)
├── image_processor.py    # Modul inti algoritma pengolahan citra
├── utils.py               # Fungsi pembantu (konversi, plotting)
├── requirements.txt
└── README.md
```

## Instalasi & Menjalankan

```bash
# 1. Clone repository
git clone <link-repository-anda>
cd filter_foto

# 2. (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

## Anggota Kelompok

- [Nama Anggota 1]
- [Nama Anggota 2]
- [Nama Anggota 3]
