FROM python:3.9-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .

# Perbarui pip ke versi terbaru untuk menghindari masalah kompatibilitas
RUN pip install --upgrade pip

# Instal dependensi dari requirements.txt
RUN pip install -r requirements.txt

# (Opsional) Jika menggunakan API, buka port 8000
EXPOSE 8000

# Jalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
