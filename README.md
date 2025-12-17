# TA-10: Simulasi Sistem Kaku (Stiff ODE) – Reaksi Kimia Robertson

Aplikasi ini merupakan implementasi Tugas Akhir 10 (TA-10) mata kuliah
Metode Numerik / Modeling & Simulation, yang membahas simulasi
**sistem persamaan diferensial kaku (stiff ODE)** menggunakan
studi kasus **reaksi kimia Robertson**.

Aplikasi dikembangkan menggunakan **Python** dan dideploy sebagai
**web app interaktif** menggunakan **Streamlit**.

---

## 👤 Identitas
- **Nama** : Roy Bakti Surya Medal  
- **NIM**  : 301230061  
- **Kelas**: IF-5B  

---

## 📌 Fitur Aplikasi
Aplikasi Streamlit ini menampilkan:
- Simulasi sistem ODE reaksi kimia Robertson
- Bukti kegagalan metode eksplisit (Euler)
- Perbandingan solver:
  - Euler eksplisit
  - RK45 (eksplisit adaptif)
  - Radau dan BDF (solver stiff)
- Visualisasi grafik hasil simulasi
- Tabel kinerja (waktu komputasi dan jumlah evaluasi fungsi)

---

## 🧪 Metode Numerik yang Digunakan
- Euler eksplisit
- Runge–Kutta orde 4/5 (RK45)
- Radau
- Backward Differentiation Formula (BDF)

---

## 🛠️ Teknologi yang Digunakan
- Python 3
- NumPy
- SciPy
- Matplotlib
- Pandas
- Streamlit
