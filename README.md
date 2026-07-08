# 🛸 AI Academic Advisor & Biometric Security Gate

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/PostgreSQL-Database-blue?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Three.js-3D_Engine-lightgrey?style=for-the-badge&logo=three.js&logoColor=black" alt="Three.js">
</p>

---

## 📌 Deskripsi Proyek
Sistem Cerdas Informasi Perpustakaan ini merupakan inovasi sistem gerbang keamanan biometrik dan penasihat akademik berbasis **Kiosk Digital**. Sistem ini mengintegrasikan kecerdasan buatan untuk otentikasi presensi mahasiswa sekaligus menyajikan antarmuka visual **3D Real-time** yang sangat interaktif dan modern.

> ✨ **Fitur Unggulan:** Antarmuka sinematik selayar penuh (*Full-Screen*) yang menghidupkan karakter robot ikonik **Wall-E & EVE** dalam bentuk grafis 3D interaktif bertenaga WebGL (Three.js), bergerak dinamis melintasi layar untuk memandu interaksi pengguna.

---

## 🚀 Fitur Utama Sistem

### 1. 👁️ Otentikasi Biometrik & Anti-Spoofing
* **Face Recognition:** Mengenali wajah mahasiswa secara akurat menggunakan pustaka berbasis *deep learning*.
* **Liveness Detection (Anti-Spoofing):** Mencegah kecurangan presensi menggunakan foto/video tiruan dengan mendeteksi frekuensi kedipan mata (*blink rate detection*) secara *real-time*.

### 2. 🌌 Antarmuka Sinematik 3D (Wall-E & EVE)
* **Mode Standby:** Robot Wall-E dan EVE versi 3D terbang bebas melintasi layar laptop secara dinamis untuk menyapa pengguna.
* **Mode Success:** Begitu wajah berhasil diverifikasi, material robot otomatis berubah warna menjadi hijau neon dan melakukan gerakan selebrasi gembira.

### 3. 📊 Dashboard Data Mahasiswa Interaktif
* **Intensitas Membaca:** Menyajikan rekapitulasi riwayat kunjungan total dan jumlah buku unik yang dibaca mahasiswa.
* **Tren Aktivitas:** Grafik garis dinamis bertenaga *Chart.js* untuk memantau perkembangan belajar.
* **Auto-Logout Keamanan:** Perlindungan sesi gerbang selama 10 detik penuh sebelum layar otomatis kembali terkunci ke mode siaga.

### 4. 📱 Integrasi Telemetri Meja (Scan QR Code)
* **Konsep BYOD (Bring Your Own Device):** Mahasiswa cukup memindai QR Code di meja baca perpus via HP untuk memasukkan judul buku fisik secara mandiri, yang otomatis terhitung ke dalam log kehadiran tanpa antre.

---

## 🛠️ Arsitektur Teknologi

| Komponen | Teknologi | Peran / Fungsi |
| :--- | :--- | :--- |
| **Backend Core** | Python & Flask | Pemrosesan logika sistem, routing, dan integrasi modul AI. |
| **AI Processing** | OpenCV, Face Recognition, Dlib | Deteksi wajah, ekstraksi landmark 68 titik, dan analitik kedipan. |
| **Database** | PostgreSQL & SQLAlchemy | Penyimpanan relasional data mahasiswa, log kunjungan, dan buku. |
| **3D Graphics** | WebGL & Three.js | Rendering dan animasi realtime objek 3D Wall-E & EVE di browser. |
| **Data Analytics** | Chart.js | Visualisasi statistik tren membaca pribadi mahasiswa. |

---

## 📦 Panduan Instalasi & Penggunaan

### 1. Prasyarat Sistem
Pastikan laptop kamu sudah terpasang:
* Python 3.9 atau versi di atasnya
* PostgreSQL Database Server

### 2. Pemasangan Pustaka Dependencies
Buka terminal pada direktori proyek, lalu pasang pustaka yang diperlukan:
```bash
pip install flask flask_sqlalchemy psycopg2 opencv-python face_recognition numpy dlib scikit-learn pyttsx3 scipy
