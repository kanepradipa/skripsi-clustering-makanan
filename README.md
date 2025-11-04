# Klasterisasi Makanan (Skripsi)

Repositori ini berisi implementasi sederhana untuk mengelompokkan (clustering) makanan berdasarkan nilai gizi menggunakan dua metode: K-means dan DBSCAN.

Struktur singkat:

- `backend/` : Flask API yang menangani parsing file (CSV/Excel), menjalankan algoritma clustering, dan menghitung metrik.
- `frontend/` : Streamlit UI untuk upload file, memilih atribut, normalisasi, menjalankan clustering, dan menampilkan hasil/visualisasi.
- `run-all.bat`, `run-all.sh` : Script untuk menjalankan backend + frontend secara otomatis (Windows / Linux-Mac).
- `run-all.ps1` : (PowerShell) script untuk menjalankan backend + frontend di jendela terpisah.

## Ringkasan cepat

- Backend: `backend/app.py` (Flask)
- Core algorithms: `backend/clustering.py`
- Frontend: `frontend/app.py` (Streamlit)

## Persyaratan

- Python 3.8+
- pip

Dependencies ada di:
- `backend/requirements.txt`
- `frontend/requirements.txt`

Direkomendasikan: gunakan virtual environment terpisah untuk backend dan frontend.

## Cara menjalankan (Windows PowerShell)

1) Clone / buka folder project.

2) Jalankan script otomatis (Windows):

   - Jika ingin menggunakan batch (CMD) yang sudah ada, cukup jalankan `run-all.bat` (double-click atau dari CMD):

     - `run-all.bat` membuat/aktifkan `venv` (root project) lalu membuka dua jendela CMD: backend dan frontend.

   - Jika ingin memakai PowerShell (direkomendasikan karena kontrol lebih baik), gunakan `run-all.ps1` yang disertakan. Untuk menjalankannya dari PowerShell (jalankan sebagai Administrator bila diperlukan untuk policy):

```powershell
# Jika policy ExecutionPolicy membatasi, Anda bisa menjalankan (sekali) di PowerShell sebagai Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Kemudian jalankan script (di root project):
.\run-all.ps1
```

3) Alternatif manual (dua terminal terpisah):

   Terminal A (backend):

```powershell
# buat venv di root (hanya sekali)
python -m venv venv
.\venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
python app.py
```

   Terminal B (frontend):

```powershell
# buat venv terpisah (opsional) atau gunakan venv root
python -m venv .venv_frontend
.\.venv_frontend\Scripts\Activate.ps1
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

4) Buka browser ke `http://localhost:8501` (Streamlit). Backend berjalan di `http://localhost:5000`.

## Penjelasan singkat run-all scripts

- `run-all.bat` - Windows batch yang sudah ada; memanggil `venv\Scripts\activate.bat` dan membuka dua jendela CMD.
- `run-all.sh` - script bash untuk Linux/Mac (sudah ada).
- `run-all.ps1` - PowerShell script (ditambahkan di repo ini) yang membuka dua jendela PowerShell, memastikan virtual environment (`venv`) ada, lalu menjalankan backend dan frontend.

Catatan: script PowerShell membuka jendela baru untuk tiap service. Jika ExecutionPolicy membatasi script, ubah scope CurrentUser seperti contoh di atas.

## Troubleshooting singkat

- Jika frontend tidak dapat terhubung: pastikan backend sudah berjalan di port 5000.
- Untuk error upload CSV/Excel: periksa format file, encoding, dan pastikan ada kolom numerik.
- Jika PCA/plot tidak muncul, pastikan `scikit-learn` terinstal di environment frontend (PCA digunakan untuk visualisasi di Evaluation page).