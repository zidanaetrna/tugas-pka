# Pengaruh Penanganan Missing Values terhadap Akurasi KNN + Grid Search

Kode eksperimen untuk makalah PKA Kelas D: dataset diabetes (767 baris), KNN yang dioptimasi dengan
Grid Search, delapan skenario penanganan missing values, nested cross-validation 10-fold x 3 ulangan,
hold-out 80:20, corrected resampled t-test, dan RMSE imputasi.

Lingkungannya sama dengan makalah: Python 3.8, scikit-learn 1.3.2, pandas 2.0.3, NumPy 1.24.4,
SciPy 1.10.1, Matplotlib 3.7.5.

## Menjalankan dengan Docker

Yang dibutuhkan hanya Docker (Docker Desktop di Windows/macOS). Image tersedia di Docker Hub:
[`itsaetrna/pka-knn-missing-values`](https://hub.docker.com/r/itsaetrna/pka-knn-missing-values)
(cerminan yang sama juga ada di `ghcr.io/zidanaetrna/pka-knn-missing-values:1.0`).

### Cara 1: docker run

```bash
docker pull itsaetrna/pka-knn-missing-values:1.0
docker run -d -p 8080:8080 --name pka-knn itsaetrna/pka-knn-missing-values:1.0
```

### Cara 2: docker compose

Dari folder repo ini (atau cukup salin file `docker-compose.yml`):

```bash
docker compose up -d
```

### Hasil

Buka **http://localhost:8080**. Selama eksperimen berjalan, halaman menampilkan progres dan memuat
ulang otomatis. Setelah selesai, halaman menampilkan tabel hasil dan grafik. Eksperimen penuh
memakan waktu sekitar 5-20 menit, tergantung jumlah inti CPU.

Dengan Cara 2, semua hasil juga tersimpan di folder `hasil/` di samping `docker-compose.yml`
(Cara 1: salin dengan `docker cp pka-knn:/app/hasil ./hasil`):

| Berkas | Isi |
|---|---|
| `index.html` | Halaman hasil |
| `tabel_performa.csv` | Akurasi, SD, selisih, p, F1, recall, AUC, hold-out per skenario |
| `tabel_grid_search.csv` | Efek Grid Search dan parameter terpilih |
| `tabel_rmse_imputasi.csv` | Galat imputasi |
| `akurasi_per_lipatan.csv` | Hasil setiap lipatan nested CV |
| `grafik_akurasi.png` | Akurasi nested CV dengan selang kepercayaan 95% |
| `hasil.json` | Semua angka dalam satu berkas |

Perintah lain:

```bash
docker logs -f pka-knn                # melihat progres di terminal
docker cp pka-knn:/app/hasil ./hasil  # menyalin berkas hasil (Cara 1)
docker stop pka-knn                   # menghentikan container (Cara 1)
docker compose down                   # menghentikan container (Cara 2)
```

Jika folder `hasil/` sudah berisi hasil, container langsung menampilkannya tanpa menjalankan ulang.
Untuk menjalankan ulang, ubah `RERUN: "0"` menjadi `RERUN: "1"` di `docker-compose.yml`, lalu
jalankan `docker compose up -d` lagi. Untuk uji cepat (3-fold x 1 ulangan, sekitar 2 menit),
isi `EKSPERIMEN_ARGS: "--quick"`.

## Menjalankan tanpa Docker

Membutuhkan Python 3.8.

```bash
pip install -r requirements.txt
python src/eksperimen.py --data data/diabetes.csv --out hasil
```

## Contoh hasil

Folder `contoh_hasil/` berisi keluaran run penuh yang angkanya sama dengan makalah (akurasi tanpa penanganan 75,40%, MICE 75,58%, hold-out 74,03% dan 73,38%). Buka `contoh_hasil/index.html` untuk melihatnya tanpa menjalankan apa pun.

## Struktur

```
data/diabetes.csv        dataset dari dosen pengampu
src/eksperimen.py        seluruh eksperimen
src/laporan_html.py      halaman hasil dan halaman progres
src/serve.py             entrypoint container (menjalankan eksperimen lalu menyajikan hasil di port 8080)
contoh_hasil/            hasil run penuh sebagai pembanding
.github/workflows/       GitHub Actions untuk build dan push image
Dockerfile
docker-compose.yml
requirements.txt
```

## Untuk kelompok: build dan push image

Image di-build dan di-push otomatis oleh GitHub Actions (`.github/workflows/docker-publish.yml`)
setiap ada push ke branch `main`, untuk platform `linux/amd64` dan `linux/arm64` (Apple Silicon),
ke Docker Hub (`itsaetrna/pka-knn-missing-values`) dan GHCR
(`ghcr.io/zidanaetrna/pka-knn-missing-values`). Bisa juga dijalankan manual dari tab
**Actions > Build dan push image Docker > Run workflow**.

Push ke Docker Hub membutuhkan secret `DOCKERHUB_TOKEN` (Personal Access Token Docker Hub dengan
akses Read & Write) di Settings > Secrets and variables > Actions. Tanpa secret itu, image hanya
di-push ke GHCR.
