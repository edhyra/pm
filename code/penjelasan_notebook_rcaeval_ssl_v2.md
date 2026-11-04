# Penjelasan Notebook: Self-Supervised Learning untuk Identifikasi Service Penyebab Gangguan (RCAEval RE1)

> Dokumen ini menjelaskan notebook `rcaeval_ssl_ts2vec_timemae_run.ipynb` dalam versi terbaru: **RE1 saja, dijalankan per sistem (OB, SS, TT) dengan arsitektur yang sama, dan kanal service non-kelas dibuang**.
> Penjelasan disusun berurutan mengikuti alur notebook. Bagian 12 menyebut apa yang sudah dan belum diuji. **Notebook belum dijalankan penuh, jadi dokumen ini tidak memuat angka hasil eksperimen.**

---

## Daftar Isi

0. [Ringkasan](#0-ringkasan)
1. [Latar Belakang dan Pertanyaan Penelitian](#1-latar-belakang-dan-pertanyaan-penelitian)
2. [Keputusan Kelompok dan Dampaknya pada Notebook](#2-keputusan-kelompok-dan-dampaknya-pada-notebook)
3. [Alur Kerja](#3-alur-kerja)
4. [Data](#4-data)
5. [Pembagian Data (Split)](#5-pembagian-data-split)
6. [Window dan Representasi Case](#6-window-dan-representasi-case)
7. [Model](#7-model)
8. [Evaluasi](#8-evaluasi)
9. [Konfigurasi](#9-konfigurasi)
10. [Cara Eksekusi](#10-cara-eksekusi)
11. [Keluaran dan Kesimpulan Otomatis](#11-keluaran-dan-kesimpulan-otomatis)
12. [Status dan Pengujian](#12-status-dan-pengujian)
13. [Keterbatasan dan Catatan Metodologis](#13-keterbatasan-dan-catatan-metodologis)
14. [Langkah Lanjut](#14-langkah-lanjut)
15. [Lampiran](#15-lampiran)

---

## 0. Ringkasan

- **Tujuan.** Menentukan *service* mana yang menjadi akar masalah dari pola metrik deret waktu, dengan memakai representasi hasil *self-supervised learning* (SSL) dan sedikit label.
- **Data.** Dataset RCAEval generasi **RE1**, tiga sistem: Online Boutique (OB), Sock Shop (SS), dan Train Ticket (TT). Tiap sistem punya 5 kelas service.
- **Cara kerja.** Tiap sistem diproses **terpisah** dengan arsitektur dan hyperparameter yang sama. Tiga keluarga SSL dibandingkan: TS2Vec, TimeMAE-mini, dan RUN-adapted. Semuanya dievaluasi dengan protokol yang sama (*linear probe*).
- **Pembanding.** Fitur statistik, encoder tanpa pretraining, model supervised dari nol, dan RUN asli (opsional).
- **Pembersihan.** Kanal metrik milik service yang bukan kelas (mis. `frontend`) dibuang, dan case yang rusak dibuang.
- **Status.** Kode lengkap dan alurnya sudah diuji, tetapi belum ada hasil eksperimen penuh (lihat Bagian 12).

---

## 1. Latar Belakang dan Pertanyaan Penelitian

### 1.1 Masalah

Aplikasi modern terdiri dari banyak *microservice*. Ketika satu service bermasalah (CPU penuh, memori bocor, jaringan lambat), dampaknya merambat ke service lain sehingga banyak service terlihat bermasalah sekaligus. Menemukan service yang menjadi **sumber** gangguan disebut ***Root Cause Analysis* (RCA)**.

Tiap service menghasilkan banyak metrik berupa deret waktu (CPU, memori, latensi, dan seterusnya). Dalam notebook ini RCA diperlakukan sebagai **klasifikasi**: dari pola metrik satu insiden, tebak service penyebabnya.

### 1.2 Mengapa SSL

Label akar masalah mahal karena harus diselidiki manusia, sedangkan data metrik tanpa label melimpah. SSL belajar dari data tanpa label lewat tugas buatan (*pretext task*), misalnya menebak bagian sinyal yang disembunyikan. Hasilnya adalah **embedding**, yaitu vektor angka yang merangkum isi sebuah potongan deret waktu. Embedding itu lalu dipakai dengan sedikit label.

### 1.3 Pertanyaan yang Dijawab

1. Apakah embedding SSL lebih baik daripada fitur statistik sederhana dan daripada encoder yang tidak dilatih?
2. Apakah keunggulan SSL terlihat saat label **sedikit** (*label efficiency*)?
3. Bagaimana SSL dibandingkan model supervised yang dilatih dari nol?
4. Seberapa baik model pada **jenis fault yang tidak pernah dilihat** saat training?
5. Apakah kesimpulan konsisten pada tiga sistem yang berbeda (OB, SS, TT)?

### 1.4 Tiga Keluarga SSL

| Keluarga | Gagasan | Sumber implementasi |
|---|---|---|
| **TS2Vec** | *Contrastive learning* hierarkis pada deret waktu | Repo resmi `yuezhihan/ts2vec` |
| **TimeMAE** | *Masked autoencoder*: sebagian potongan sinyal disembunyikan, model menebak representasinya | **Re-implementasi ringkas** (`TimeMAEMini`), bukan repo resmi |
| **RUN** | *Contrastive learning* tanpa pasangan negatif sebagai encoder, lalu *causal discovery* | **Adaptasi** (`ContrastiveNoNegMini`), hanya tahap encoder |

---

## 2. Keputusan Kelompok dan Dampaknya pada Notebook

| Keputusan | Wujud di notebook |
|---|---|
| Data dibagi **per sistem**, arsitektur sama | `CFG["SYSTEMS"]` dan fungsi `run_all_systems`. Tiap sistem dimuat ulang dan dilatih dari awal. Hyperparameter model satu set. Hanya jumlah kanal input yang menyesuaikan sistemnya |
| **Hanya RE1** | `SYSTEMS = ["RE1-OB", "RE1-SS", "RE1-TT"]`, dengan `assert` bahwa semua nama berawalan `RE1-`. RE2 dan RE3 di luar cakupan |
| Kelas yang tidak ada di RE1 (mis. `frontend`) dibuang | Label sudah bersih (lihat 4.4). Yang dibuang adalah **kanal metrik** milik service non-kelas, diatur `PRUNE_CHANNELS` |

Satu langkah tambahan **di luar keputusan kelompok**: case yang rusak dibuang (lihat 4.3). Langkah ini bisa dimatikan dengan menghapus blok "case rusak" di fungsi `load_all`.

---

## 3. Alur Kerja

```
Untuk setiap sistem RE1 (OB, SS, TT):
        │
        ▼
[1] Muat data, buang case rusak, buang kanal non-kelas, z-score
        │
        ▼
[2] Split anti-bocor ── easy / medium / hard
        │
        ├────────────► [3] Pretraining SSL (TANPA label, hanya case train)
        │                      TS2Vec · TimeMAE-mini · RUN-adapted
        ▼                              │
[4] Ambil window normal & window faulty tiap case
        │                              │
        └──────────────┬───────────────┘
                       ▼
[5] Embedding case = rata-rata embedding window faulty − window normal
                       │
                       ▼
[6] Linear probe (Logistic Regression) → prediksi service penyebab
                       │
                       ▼
[7] Metrik: accuracy · macro-F1 · Top-3  ·  kurva label-efficiency
        │
        ▼
Gabungkan tiga sistem → tabel per sistem, tabel lintas sistem, kesimpulan otomatis
```

Prinsipnya: **semua model memakai prosedur yang sama** (preprocessing, split, dan probe identik). Dengan begitu selisih skor mencerminkan kualitas representasi, bukan perbedaan prosedur.

---

## 4. Data

### 4.1 Dataset

**RCAEval** (`phamquiluan/RCAEval` di Hugging Face). Satu *case* adalah satu insiden: metrik deret waktu (disampling 1 detik), waktu penyuntikan fault (`inject_time`), jenis fault, nomor repetisi, dan label `root_cause_service`. Indeks seluruh case ada di `cases.parquet`.

### 4.2 Tiga Sistem RE1

Tiap sistem berisi 125 case: **5 service penyebab × 5 jenis fault × 5 repetisi**. Jenis fault: `cpu`, `delay`, `disk`, `loss`, `mem`.

| Sistem | Case dipakai | Kelas (5 per sistem) | Kanal gabungan | Kanal dipakai |
|---|---|---|---|---|
| RE1-OB (Online Boutique) | 124 (dari 125) | adservice, cartservice, checkoutservice, currencyservice, productcatalogservice | 86 | 39 |
| RE1-SS (Sock Shop) | 125 | carts, catalogue, orders, payment, user | 63 | 30 |
| RE1-TT (Train Ticket) | 125 | ts-auth-service, ts-order-service, ts-route-service, ts-train-service, ts-travel-service | 241 | 30 |

Angka kanal diperoleh dengan menjalankan fungsi pemuatan data pada data asli. Nama service pada tabel adalah prefix nama metrik (mis. `adservice_cpu` → `adservice`).

### 4.3 Pembersihan Data

| Masalah pada data | Penanganan | Fungsi |
|---|---|---|
| Jumlah metrik berbeda antar case (mis. 49–59 di RE1-OB) | Kolom disatukan (*union*), kolom yang tidak ada di suatu case diisi **0** setelah normalisasi | `load_all`, `preprocess_case` |
| `inject_time` rusak atau di luar rentang waktu seri (mis. `re1ob_currencyservice_loss_1` = 16.933.142) | Diganti **titik tengah seri**, karena RE1 menyuntik fault di tengah | `fix_inject` |
| Kolom `time` ganda pada sebagian case RE1-OB | Kolom duplikat dibuang | `read_case` |
| Nilai tak hingga atau kosong | `inf` menjadi `NaN`, lalu diisi `ffill` dan `bfill` | `preprocess_case` |
| Skala metrik sangat berbeda | **Z-score** per case (lihat 4.5) | `preprocess_case` |
| **Case rusak**: seri lebih pendek dari `WIN` (128) atau tanpa periode faulty | Case dibuang dan namanya dicetak. Kasus nyata: `re1ob_productcatalogservice_cpu_3` (63 detik, 0 titik faulty). Akibatnya `productcatalogservice`/`cpu` di OB tinggal 4 repetisi | `load_all` |

Filter case rusak memakai kolom `n_timesteps` dan `faulty_timesteps` dari indeks. Jika kolom itu tidak ada (data sintetis), filter dilewati.

### 4.4 Pembuangan Kanal Non-Kelas

**Pengecekan kelas.** Kelas diambil dari `root_cause_service` pada indeks RE1, sehingga tiap sistem otomatis hanya punya 5 kelas. Service seperti `frontend` hanya menjadi kelas di RE3-SS, sedangkan `emailservice` dan `recommendationservice` hanya di RE2/RE3. Jadi **tidak ada label yang perlu dibuang**.

**Yang dibuang adalah kanal metriknya.** Service tersebut tetap punya metrik di data RE1 walaupun tidak pernah menjadi akar masalah. Aturannya: sebuah kanal dipertahankan hanya jika prefix service-nya (bagian nama sebelum `_` pertama) **sama persis** dengan salah satu kelas sistem itu.

| Sistem | Service yang kanalnya dibuang |
|---|---|
| OB (10 service) | `frontend`, `frontend-check`, `frontend-external`, `emailservice`, `paymentservice`, `recommendationservice`, `shippingservice`, `redis`, `main`, `PassthroughCluster` |
| SS (11 service) | `front-end`, `shipping`, `queue-master`, `rabbitmq`, `rabbitmq-exporter`, `session-db`, `istio-init`, `carts-db`, `catalogue-db`, `orders-db`, `user-db` |
| TT (59 service) | Semua service selain lima kelas, termasuk `ts-ui-dashboard` dan seluruh `ts-*-mongo` |

Konsekuensinya, kanal DB pendamping seperti `carts-db` dan `ts-auth-mongo` ikut terbuang karena prefix-nya bukan kelas. Kode memastikan tiap kelas masih punya minimal satu kanal (`assert`). Pembuangan bisa dimatikan dengan `PRUNE_CHANNELS=False`.

**Catatan kualitas kanal (hasil pemeriksaan data, tidak diubah oleh notebook):**

| Sistem | Kanal dipakai | Ada di semua case | Hanya di sebagian case |
|---|---|---|---|
| OB | 39 | 10 | 29 (14 di antaranya ada di kurang dari separuh case) |
| SS | 30 | 26 | 4 |
| TT | 30 | 25 | 5 |

Pada OB terdapat dua skema penamaan: 50 case memakai `*_latency` dan `*_load`, sedangkan 75 case memakai `*_latency-50`, `*_latency-90`, dan `*_workload`. Notebook menganggapnya kanal berbeda, sehingga tiap case OB punya banyak kanal yang terisi 0. Kolom `*_error` juga hanya ada di sebagian kecil case. SS dan TT jauh lebih seragam.

### 4.5 Normalisasi Z-score

Per kolom metrik dalam satu case:

```
z = (x − mean_normal) / std_normal ,   lalu di-clip ke [−10, +10]
```

- `mean_normal` dan `std_normal` hanya dari **periode normal**, yaitu data sebelum `inject_time`. Nilai z yang besar berarti "menyimpang dari kondisi normal case itu sendiri".
- Bila `std_normal` hampir nol, dipakai std seluruh seri. Bila masih nol, dipakai 1.
- *Clipping* ±10 mencegah satu lonjakan ekstrem mendominasi pelatihan.
- Titik injeksi dibatasi minimal 8 baris dari tepi seri agar periode normal tidak kosong.

### 4.6 Struktur Data di Memori

Hasil pemuatan satu sistem disimpan di dictionary global `DATA`:

| Kunci | Isi |
|---|---|
| `Z` | Daftar array `(T, C)` per case: deret waktu ternormalisasi |
| `inj` | Indeks baris tempat fault mulai |
| `cols` | Daftar kanal yang dipakai (C kanal) |
| `y`, `classes` | Label service per case dan daftar kelas |
| `fault`, `rep` | Jenis fault dan nomor repetisi per case |
| `dropped`, `n_union` | Service yang kanalnya dibuang, dan jumlah kanal sebelum dibuang |

`DATA` selalu berisi sistem yang **sedang diproses**. Fungsi `prepare_system(name)` memuat satu sistem, mencetak ringkasannya (jumlah case, kanal, kelas, service yang dibuang, dan tabel case × fault), lalu menyimpannya ke `SYS_INFO[name]`. `SYS_INFO` dipakai sel hasil dan kesimpulan agar tidak bergantung pada `DATA` milik sistem terakhir.

### 4.7 Data Sintetis (`DRY_RUN`)

Dengan `DRY_RUN=True` notebook membuat data buatan tanpa internet: 5 service × 4 fault × 5 repetisi = 100 case, 6 kanal per service, panjang 721 detik. Fault disimulasikan sebagai gelombang sinus pada kanal service penyebab setelah titik tengah. Data ini memuat dua jebakan untuk menguji pipeline: satu case dengan `inject_time` rusak, dan fault `delay` yang kehilangan satu kolom. Tujuannya **hanya uji alur**, bukan menghasilkan angka untuk laporan. Mode ini berjalan sebagai satu sistem bernama `SYN`.

---

## 5. Pembagian Data (Split)

### 5.1 Mengapa Tidak Acak

Lima repetisi untuk tiap pasangan (service, fault) hampir identik. Bila dibagi acak ke train dan test, model bisa "menghafal" case yang hampir sama dan skornya tampak tinggi secara semu. Ini disebut **kebocoran data** (*data leakage*).

### 5.2 Tiga Skema (`make_splits`)

| Skema | Cara membagi | Ukuran (tiap sistem RE1) | Yang diukur |
|---|---|---|---|
| **`easy`** | 5-fold stratified per service | 5 fold | Skenario paling longgar. Repetisi dari pasangan yang sama bisa muncul di train dan test |
| **`medium`** (bawaan) | Train = repetisi 1–3, test = repetisi 4–5 | Sekitar 75 train, 50 test | Generalisasi ke repetisi yang belum terlihat |
| **`hard`** | *Leave-one-fault-type-out*: train pada empat jenis fault, test pada jenis yang ditahan | 5 fold, tiap fold test 25 case (24 pada fold `cpu` di OB) | Generalisasi ke **jenis fault baru**. Ukuran paling jujur |

Batas `medium` dihitung dari `ceil(repetisi_maks × 0,6) = 3`. Pada OB, case yang dibuang berasal dari repetisi 3, sehingga train menjadi 74.

### 5.3 Pretraining Tidak Boleh Melihat Test

Pretraining SSL tidak memakai label, tetapi tetap hanya memakai window dari **case train** pada split yang sedang berjalan, supaya informasi data test tidak ikut masuk.

---

## 6. Window dan Representasi Case

### 6.1 Window

Model menerima potongan deret waktu berukuran tetap (*window*), berbentuk `(N, WIN, C)` dengan `WIN = 128` detik.

| Jenis | Cara mengambil | Fungsi |
|---|---|---|
| **Pretraining** | Titik awal acak di seluruh seri case train (`pretrain_windows`), sebanyak `PRE_WIN` per case | Data latih SSL tanpa label |
| **Normal** | `EVAL_WIN` window berjarak merata dari awal seri sampai sebelum `inject_time` | Kondisi sehat case |
| **Faulty** | `EVAL_WIN` window berjarak merata dari `inject_time` sampai akhir seri | Kondisi saat gangguan |

Bila seri terlalu pendek untuk window penuh, window diperpanjang dengan menyalin nilai tepi (`np.pad` mode `edge`).

### 6.2 Representasi Case

Satu case diwakili satu vektor:

```
R(case) = rata-rata embedding window faulty − rata-rata embedding window normal
```

Yang berguna untuk RCA adalah **perubahan** akibat fault, bukan kondisi dasar sistem. Pengurangan ini menghilangkan ciri khas case yang tak terkait gangguan. Mode lain lewat `CASE_REPR`: `fault` (hanya faulty) dan `concat` (gabungan faulty dan selisih).

---

## 7. Model

Semua embedder SSL punya antarmuka sama: `fit(windows)` untuk pretraining, lalu `encode(windows)` yang menghasilkan matriks `(n, d)`.

| Nama di `CFG["MODELS"]` | Jenis | Label dipakai? | Pretraining? |
|---|---|---|---|
| `stat` | Baseline fitur statistik | Hanya di probe | Tidak |
| `random_ts2vec` | Kontrol: TS2Vec tanpa pelatihan | Hanya di probe | Tidak |
| `ts2vec` | SSL contrastive hierarkis | Hanya di probe | Ya |
| `timemae_mini` | SSL masked autoencoder | Hanya di probe | Ya |
| `runcl_mini` | SSL contrastive tanpa negative pair | Hanya di probe | Ya |
| `sup_conv` | Supervised dari nol | **Ya**, end-to-end | Tidak |
| RUN-native (opsional) | RUN asli dari paket RCAEval | Tidak | Tidak (dilatih per case) |

### 7.1 Baseline Statistik (`stat`)

Tanpa jaringan saraf. Pada bagian faulty tiap case (setelah `inject_time`) dihitung per kanal: rata-rata z-score, nilai absolut maksimum, dan simpangan baku. Hasilnya vektor 3 × C. Baseline ini menjawab: *apakah SSL lebih baik daripada ringkasan sederhana?*

### 7.2 TS2Vec dan Kontrol `random_ts2vec`

- **Cara kerja.** Dua potongan dari deret waktu yang sama harus punya embedding mirip, dan potongan dari deret waktu lain harus berbeda. Kontras dilakukan **hierarkis** di berbagai skala waktu, baik antar-timestamp maupun antar-instance.
- **Konfigurasi.** `output_dims=320`, `hidden_dims=64`, `depth=10`, `batch_size=16`. Embedding diambil dengan `encoding_window="full_series"`, sehingga satu window menjadi satu vektor.
- **Kontrol `random_ts2vec`.** Arsitektur yang sama persis tetapi **tidak dilatih**. Bila `ts2vec` tidak mengalahkannya, pretraining tidak menambah nilai.

### 7.3 TimeMAE-mini (`TimeMAEMini`)

Re-implementasi ringkas gaya TimeMAE, **bukan** kode resmi. Model diminta menebak bagian sinyal yang disembunyikan, sehingga ia harus memahami pola waktunya.

1. **Patchify.** Window 128 detik dipotong menjadi 16 patch berukuran 8 detik (`PATCH=8`). Tiap patch meratakan 8 langkah waktu × C kanal.
2. **Masking.** 60% patch (10 dari 16) disembunyikan acak, 6 patch terlihat.
3. **Encoder terdekoupel.** Encoder Transformer (dimensi 128, 3 lapis, 4 head) hanya melihat **patch yang terlihat**. Token topeng tidak masuk encoder.
4. **Decoder.** Decoder Transformer 1 lapis menerima representasi patch terlihat ditambah *mask token* di posisi tersembunyi, lalu memprediksi representasi posisi itu.
5. **Target dari encoder EMA.** Target dihasilkan salinan encoder (*target encoder*) yang melihat **sinyal utuh**. Parameternya tidak dilatih dengan gradien, tetapi mengikuti encoder utama lewat *Exponential Moving Average* (`ema=0.99`).

| Loss | Arti | Cara kerja |
|---|---|---|
| **MRR** (*Masked Representation Regression*) | Menebak nilai representasi | MSE antara prediksi dan representasi target di posisi tersembunyi |
| **MCC** (*Masked Codeword Classification*) | Menebak "kata kode" | Representasi target dipetakan ke codeword terdekat (cosine) dalam codebook 128 kode. Model mengklasifikasi codeword itu dengan cross-entropy. Codebook diperbarui dengan EMA |

Loss total = `MRR + 0,5 × MCC`. Optimasi memakai AdamW (lr 5e-4, weight decay 1e-4), jadwal `OneCycleLR`, *gradient clipping* 1,0, dan EMA diperbarui tiap langkah. **Encoding:** encoder utama melihat semua patch, hasilnya dirata-ratakan sepanjang waktu menjadi vektor 128 dimensi.

### 7.4 RUN-adapted (`ContrastiveNoNegMini`)

RUN asli memakai *contrastive learning* tanpa pasangan negatif untuk memperkaya encoder, lalu melanjutkan dengan *neural Granger causal discovery* dan *PageRank*. Notebook hanya mengambil **tahap encoder** agar bisa dibandingkan dalam protokol yang sama. Tahap kausal tidak ada. Augmentasi dan arsitekturnya adalah pilihan penulis notebook, bukan kode asli RUN.

**Cara kerja (gaya SimSiam):**

1. Dari satu batch dibuat **dua view** dengan augmentasi berbeda: *crop* acak 75% panjang window, perkalian skala acak per kanal, *jitter* (derau Gaussian 0,05), *time masking* (15% titik waktu dinolkan), dan *channel dropout* (10% kanal dinolkan).
2. Tiap view melewati encoder `f`, lalu *projector* `g`. Hasilnya dilewatkan ke *predictor* `h`.
3. Loss adalah **negatif cosine similarity** simetris antara prediksi satu view dan proyeksi view lainnya.
4. ***Stop-gradient*** dipasang pada sisi target. Tanpa pasangan negatif, mekanisme ini bersama predictor mencegah *collapse*, yaitu semua input menghasilkan vektor yang sama.

**Encoder `_ConvEnc`:** konvolusi 1D dengan 4 blok residual berdilasi (dilasi 1, 2, 4, 8; kernel 5; BatchNorm; GELU). Keluarannya diagregasi dengan *mean pooling* dan *max pooling* sepanjang waktu, lalu diproyeksikan ke 128 dimensi. Saat encoding hanya `f` yang dipakai. Optimasi: AdamW (lr 1e-3, weight decay 1e-4).

### 7.5 Pembanding Supervised (`sup_conv`)

- Memakai encoder `_ConvEnc` yang **sama** dengan RUN-adapted, ditambah satu lapisan linear untuk klasifikasi.
- Dilatih **dengan label** end-to-end pada window faulty (24 window per case train), memakai cross-entropy, derau Gaussian 0,05 sebagai augmentasi, AdamW (lr 1e-3, weight decay 1e-2).
- Prediksi satu case = rata-rata probabilitas softmax dari window faulty case itu.
- Dilatih ulang untuk **setiap** pecahan label, tanpa pretraining.
- Perannya: *saat label sedikit, apakah SSL lebih baik daripada langsung melatih dengan label?*

### 7.6 RUN-native (Opsional)

Menjalankan RUN asli dari paket `RCAEval`. Perbedaannya dengan model lain:

| Aspek | Linear probe (model SSL) | RUN-native |
|---|---|---|
| Label | Dipakai untuk probe | **Tidak dipakai** |
| Keluaran | Kelas service | **Ranking metrik**, dipetakan ke service lewat prefix nama metrik |
| Pelatihan | Satu kali untuk semua case | **Per case** (lambat) |
| Metrik | accuracy, macro-F1, Top-3 | Top-1, Top-3 |

Setelan: jendela ±600 detik di sekitar `inject_time` dan maksimal 10 case uji per sistem (`RUN_MAX_CASES`). Mati secara bawaan (`RUN_NATIVE=False`) dan perlu `pip install "RCAEval[default]"`. Karena jenis keluarannya berbeda, angkanya **tidak boleh dibandingkan langsung** dengan linear probe. Bila dinyalakan, dijalankan per sistem dan memakai data mentah, bukan kanal yang sudah dipangkas.

---

## 8. Evaluasi

### 8.1 Linear Probe

Semua representasi dievaluasi dengan pengklasifikasi yang sama:

```
StandardScaler  →  LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
```

Pada *linear probe* encoder dibekukan dan hanya pengklasifikasi linear yang dilatih. Karena pengklasifikasinya sederhana, skor baik hanya mungkin bila representasinya sendiri sudah memisahkan kelas. `class_weight="balanced"` menyeimbangkan kelas yang jumlah case-nya tidak sama.

### 8.2 Metrik

| Metrik | Arti |
|---|---|
| **Accuracy** | Persentase case yang service-nya ditebak benar |
| **Macro-F1** | Rata-rata F1 per kelas, tiap kelas berbobot sama |
| **Top-3 accuracy** | Persentase case yang service sebenarnya ada di tiga prediksi teratas |

Akurasi acak (*chance*) = `1 / jumlah kelas` = 0,2 untuk ketiga sistem, ditampilkan sebagai garis pembanding.

### 8.3 Label Efficiency

Probe dilatih dengan `LABEL_FRACS = [0.1, 0.2, 0.5, 1.0]` dari label train. Pengambilan sampel dilakukan **per kelas** dengan minimal 1 case per kelas, sehingga semua kelas tetap terwakili. Pecahan 1% tidak dipakai karena data terlalu kecil.

- **Label penuh** mengukur **kualitas representasi**.
- **Label sedikit** mengukur **manfaat SSL**.

### 8.4 Akurasi per Jenis Fault

Pada label penuh, tiap case uji dicatat benar atau salah, lalu dirata-ratakan per (sistem, model, jenis fault) untuk melihat jenis fault yang sulit dikenali.

---

## 9. Konfigurasi

Seluruh pengaturan ada di dictionary `CFG` (sel 3).

### 9.1 Parameter Utama

| Parameter | Nilai bawaan | Arti |
|---|---|---|
| `SYSTEMS` | `RE1-OB`, `RE1-SS`, `RE1-TT` | Sistem yang dijalankan satu per satu (harus berawalan `RE1-`) |
| `DATASET` | `None` | Sistem yang sedang diproses, diisi otomatis oleh `prepare_system` |
| `DRY_RUN` | `False` | `True` = data sintetis tanpa internet |
| `QUICK` | `True` | `True` = skala kecil untuk uji cepat |
| `SPLIT` | `medium` | `easy`, `medium`, atau `hard` |
| `CASE_REPR` | `diff` | Cara membentuk representasi case |
| `MODELS` | `stat`, `random_ts2vec`, `ts2vec`, `timemae_mini`, `runcl_mini`, `sup_conv` | Model yang dijalankan |
| `LABEL_FRACS` | 0,1 · 0,2 · 0,5 · 1,0 | Pecahan label untuk probe |
| `WIN` | 128 | Panjang window (detik) |
| `PATCH` | 8 | Ukuran patch TimeMAE-mini (harus membagi `WIN`) |
| `CLIP` | 10,0 | Batas clipping z-score |
| `PRUNE_CHANNELS` | `True` | Buang kanal service non-kelas |

Hyperparameter model satu set untuk semua sistem.

### 9.2 Mode `QUICK` dan Mode Penuh

| Parameter | `QUICK=True` (bawaan) | `QUICK=False` |
|---|---|---|
| Seed (`SEEDS`) | 1 (`[0]`) | 3 (`[0, 1, 2]`) |
| Window pretraining per case (`PRE_WIN`) | 24 | 80 |
| Window evaluasi per case (`EVAL_WIN`) | 8 | 16 |
| Epoch TS2Vec | 3 | 20 |
| Epoch TimeMAE-mini | 15 | 60 |
| Epoch RUN-adapted | 15 | 60 |
| Epoch supervised | 15 | 60 |

Mode `QUICK` hanya untuk memastikan pipeline berjalan. Hasil yang dilaporkan sebaiknya memakai mode penuh dengan minimal 3 seed.

### 9.3 Reprodusibilitas

`set_seed` mengatur seed `random`, `numpy`, dan `torch` (termasuk CUDA) sebelum membangun dan melatih tiap model.

---

## 10. Cara Eksekusi

### 10.1 Urutan Sel

Sel dijalankan berurutan dari atas karena berbagi variabel global.

| Sel | Isi | Catatan |
|---|---|---|
| 2 | Install paket dan clone repo TS2Vec | Sekali saja |
| 3 | Import dan `CFG` | Ubah `QUICK`, `SPLIT`, `MODELS` di sini |
| 5 | Fungsi data, `load_all`, `prepare_system` | Memuat **sistem pertama** sebagai pratinjau |
| 7, 9 | Split, windowing, evaluasi, baseline statistik | Memakai `DATA` sistem pertama untuk mencetak ringkasan |
| 11, 13, 15 | Definisi model | Hanya mendefinisikan kelas |
| 17 | `run_all_systems` | **Memuat ulang dan melatih tiap sistem**. Bagian terlama |
| 19 | Tabel, grafik, dan tabel lintas sistem | |
| 21 | RUN-native | Dilewati kecuali `RUN_NATIVE=True` |
| 23 | Kesimpulan otomatis per sistem | |

### 10.2 Perulangan Eksperimen

```
untuk setiap sistem di CFG["SYSTEMS"]:               # run_all_systems
  prepare_system(sistem)                             # muat data, buang kanal & case rusak
  untuk setiap seed:                                 # run_experiments
    untuk setiap split (sesuai CFG["SPLIT"]):
      untuk setiap model di CFG["MODELS"]:
        jika model == "stat":
            R = fitur statistik
        jika model == "sup_conv":
            untuk setiap pecahan label: latih dari nol → evaluasi → catat
            (lanjut ke model berikutnya)
        selain itu (model SSL):
            1. ambil window pretraining dari case train (sekali per split, dipakai bersama)
            2. latih model SSL tanpa label
            3. R = embedding semua case (faulty − normal)
        untuk setiap pecahan label:
            linear probe pada R → catat accuracy, macro-F1, Top-3
```

Hal penting:

- **Data pretraining dipakai bersama** oleh semua model SSL dalam satu split, sehingga perbandingan adil.
- **Tiap sistem dilatih dari awal.** Tidak ada bobot yang dibawa antar sistem.
- **Waktu jalan** kira-kira tiga kali lipat dibanding satu sistem. Skema `easy` dan `hard` masing-masing menghasilkan 5 fold sehingga memperpanjang waktu lagi.
- **Penanganan kegagalan.** Model yang gagal (mis. repo TS2Vec belum di-clone) mencetak `[GAGAL]` dan dilewati, tidak menghentikan eksperimen.
- **Hasil** dikumpulkan dalam dua DataFrame: `results` (sistem, model, split, seed, pecahan label, jumlah case train, metrik) dan `per_fault` (sistem, jenis fault, benar atau salah per case).

---

## 11. Keluaran dan Kesimpulan Otomatis

### 11.1 Tabel dan Grafik (Sel 19)

Untuk **setiap sistem**:

- Tabel mean dan std (antar seed) untuk accuracy, macro-F1, dan Top-3 per (model, pecahan label).
- **Grafik kiri:** kurva *label efficiency* dengan error bar dan garis *chance*.
- **Grafik kanan:** diagram batang tiga metrik pada label penuh.
- Tabel akurasi per jenis fault.

Setelah itu ada **tabel lintas sistem**: accuracy dan macro-F1 pada label penuh, baris = model dan kolom = sistem.

### 11.2 Kesimpulan Otomatis (Sel 23)

`auto_conclusion(results, system, run_native_df)` dipanggil **sekali per sistem**. Tidak ada angka yang ditulis manual. Isinya:

1. Model terbaik pada label penuh dan pada label paling sedikit.
2. Perbandingan tiap model SSL dengan `random_ts2vec` dan `stat` pada label paling sedikit dan paling banyak. Selisih rata-rata `d` diberi vonis:
   - **lebih baik** bila `d` positif dan lebih besar daripada std terbesar kedua model,
   - **lebih buruk** bila `d` negatif dan nilai mutlaknya melebihi std itu,
   - **tidak beda jelas** untuk kondisi lainnya.
3. Selisih tiap model SSL terhadap `sup_conv` pada label sedikit.
4. Hasil RUN-native bila dijalankan.
5. Peringatan otomatis bila jumlah seed kurang dari 3.

### 11.3 Kriteria "SSL Bermanfaat"

Pretraining dianggap bermanfaat **hanya jika ketiganya terpenuhi**:

1. mengalahkan encoder random-init berarsitektur sama (`random_ts2vec`),
2. mengalahkan baseline fitur statistik (`stat`),
3. keunggulannya terlihat pada label sedikit.

Sel markdown terakhir di notebook berisi kesimpulan akhir yang bersifat metodologis (cara membaca hasil, keterbatasan, langkah lanjut) dan **tanpa angka**.

---

## 12. Status dan Pengujian

| Aspek | Keadaan |
|---|---|
| Jumlah sel | 25 (13 markdown, 12 code) |
| Output tersimpan | **Belum ada hasil eksperimen**. Tidak ada angka accuracy, F1, atau grafik yang bisa dikutip. Output yang mungkin tersimpan hanya dari percobaan jalan yang terhenti di awal, jadi periksa isinya sebelum mengutip |
| Konfigurasi aktif | Tiga sistem RE1, `PRUNE_CHANNELS=True`, `QUICK=True`, `SPLIT="medium"`, 1 seed, `DRY_RUN=False`, `RUN_NATIVE=False` |

**Yang sudah diuji (hanya untuk memastikan kode berjalan):**

- Data sintetis dengan **keenam model**, mode `QUICK`: berjalan sampai kesimpulan otomatis.
- Data asli **ketiga sistem RE1** dengan tiga model (`stat`, `random_ts2vec`, `sup_conv`), mode `QUICK`, 1 seed: pemuatan data, tabel per sistem, tabel lintas sistem, dan kesimpulan otomatis berjalan.

**Yang belum diuji:** `ts2vec`, `timemae_mini`, dan `runcl_mini` pada data asli; mode penuh (`QUICK=False`, minimal 3 seed); split `easy` dan `hard`; RUN-native. Angka dari uji di atas tidak dicatat sebagai hasil.

Untuk laporan: bagian rancangan dan metode (Bagian 4–10) sudah bisa ditulis dari dokumen ini. Bagian hasil dan pembahasan menunggu notebook dijalankan penuh.

---

## 13. Keterbatasan dan Catatan Metodologis

### 13.1 Keterbatasan yang Dinyatakan di Notebook

- **Data kecil.** Puluhan case per kelas dengan repetisi hampir duplikat, sehingga variansi antar split besar. Gunakan minimal 3 seed.
- **Bukan implementasi resmi.** TimeMAE-mini dan RUN-adapted adalah re-implementasi atau adaptasi. Untuk klaim publikasi perlu diulang dengan repo resmi.
- **Tanpa transfer lintas sistem.** Kelas adalah service spesifik per sistem. Tiap sistem punya 5 kelas dan dijalankan terpisah, jadi angka antar sistem **tidak ekuivalen** (kelas dan jumlah kanal berbeda).
- **Kanal non-kelas dan kanal DB pendamping dibuang.** Input dipersempit ke service kandidat. Pengaruhnya belum diuji.
- **Pilihan preprocessing memengaruhi hasil**: normalisasi dengan periode normal, window 128 detik, clip ±10, pengisian kolom hilang dengan 0. Perlu ablation.

### 13.2 Catatan Tambahan dari Pembacaan Kode dan Pemeriksaan Data

1. **Std = 0 pada 1 seed.** Dengan satu seed, simpangan baku antar seed diisi 0. Ambang "1 std" lalu praktis menjadi "selisih apa pun", sehingga vonis "lebih baik" atau "lebih buruk" pada mode `QUICK` tidak bermakna. Notebook menambahkan peringatan bila seed kurang dari 3.
2. **Pembanding supervised tidak persis setara.** `sup_conv` dilatih pada window faulty saja, sedangkan model SSL memakai selisih faulty − normal.
3. **Split `easy` paling optimistis.** Repetisi hampir duplikat bisa jatuh di train dan test sekaligus. Anggap skor `easy` sebagai batas atas.
4. **Normal vs faulty bergantung `inject_time`.** `fix_inject` hanya mengoreksi nilai yang berada di luar rentang waktu seri. `inject_time` yang salah tetapi masih di dalam rentang tidak terdeteksi dan mencemari representasi selisih.
5. **Kolom hilang diisi 0.** Kanal yang tidak dimiliki suatu case tampak seperti "tidak ada penyimpangan", yang belum tentu benar.
6. **Penamaan kanal OB tidak seragam** (lihat 4.4). Hanya 10 dari 39 kanal OB yang ada di semua case. Dua skema nama dianggap kanal berbeda, sehingga banyak kanal kosong per case.
7. **Pemilihan kanal memakai himpunan kelas.** Ini memanfaatkan fakta bahwa penyebabnya pasti salah satu dari 5 service. Wajar untuk klasifikasi 5 kelas, tetapi berbeda dari RCA nyata saat semua service menjadi kandidat. Kanal gejala seperti `frontend` yang mungkin informatif ikut hilang.
8. **Baseline `stat` bisa sangat kuat.** Pada data berkelas sedikit dengan fault yang jelas, ringkasan statistik sederhana berpotensi sudah memisahkan kelas dengan baik, terutama di split `medium`. Bila demikian, split `hard` lebih membedakan model.

---

## 14. Langkah Lanjut

1. Jalankan `QUICK=False` dengan minimal 3 seed untuk `easy`, `medium`, dan `hard` pada ketiga sistem RE1.
2. Ablation pembuangan kanal: `PRUNE_CHANNELS=True` vs `False`.
3. Seragamkan nama kanal RE1-OB (`latency` dengan `latency-50`, `load` dengan `workload`) dan lihat pengaruhnya.
4. Ablation lain: panjang window, `CASE_REPR` (`diff`, `fault`, `concat`), mask ratio, dan ukuran patch.
5. Tambahkan pipeline B (embedding per service lalu ranking) agar sebanding dengan metrik RCA standar (Avg@k).
6. Ganti TimeMAE-mini dengan repo resmi dan RUN-adapted dengan RUN penuh sebagai konfirmasi.

---

## 15. Lampiran

### 15.1 Glosarium

| Istilah | Penjelasan singkat |
|---|---|
| **RCA** (*Root Cause Analysis*) | Menentukan sumber awal suatu gangguan |
| **Microservice** | Aplikasi yang dipecah menjadi banyak layanan kecil yang saling memanggil |
| **Deret waktu** (*time series*) | Data yang diukur berurutan menurut waktu |
| **Kanal** | Satu kolom metrik, mis. `adservice_cpu` |
| **SSL** | Belajar dari data tanpa label lewat tugas buatan |
| **Embedding** | Vektor angka yang merangkum isi suatu data |
| **Pretext task** | Tugas buatan yang dipakai SSL |
| **Contrastive learning** | Melatih model agar data "mirip" punya embedding dekat dan yang berbeda berjauhan |
| **Masked autoencoder** | Menyembunyikan sebagian input lalu meminta model memulihkannya |
| **EMA** (*Exponential Moving Average*) | Rata-rata bergerak yang perubahannya halus |
| **Stop-gradient** | Menahan aliran gradien pada satu cabang jaringan |
| **Collapse** | Kegagalan SSL: semua input menghasilkan embedding yang sama |
| **Linear probe** | Menguji embedding dengan pengklasifikasi linear di atas encoder yang dibekukan |
| **Label efficiency** | Seberapa baik model bekerja saat label sedikit |
| **Data leakage** | Informasi data uji "bocor" ke data latih sehingga skor tampak terlalu tinggi |
| **Z-score** | Nilai dalam satuan simpangan baku dari rata-rata |
| **Ablation** | Mengubah atau menghapus satu komponen untuk melihat pengaruhnya |
| **Top-k accuracy** | Benar bila jawaban sebenarnya ada di k prediksi teratas |

### 15.2 Peta Sel Notebook

| Sel | Jenis | Isi |
|---|---|---|
| 0 | Markdown | Judul, alur, catatan kejujuran implementasi |
| 1–2 | Markdown, Code | Setup: install paket, clone repo TS2Vec |
| 3 | Code | Import, `CFG`, `set_seed` |
| 4–5 | Markdown, Code | Data: loader, preprocessing, data sintetis, `load_all`, `prepare_system` |
| 6–7 | Markdown, Code | Split anti-bocor dan windowing |
| 8–9 | Markdown, Code | Linear probe, metrik, baseline statistik, `embed_all` |
| 10–11 | Markdown, Code | TS2Vec dan kontrol random |
| 12–13 | Markdown, Code | TimeMAE-mini |
| 14–15 | Markdown, Code | RUN-adapted dan `sup_conv` |
| 16–17 | Markdown, Code | `build_embedder`, `run_experiments`, `run_all_systems` |
| 18–19 | Markdown, Code | Tabel dan grafik per sistem, tabel lintas sistem |
| 20–21 | Markdown, Code | RUN-native (opsional, per sistem) |
| 22–23 | Markdown, Code | Kesimpulan otomatis per sistem |
| 24 | Markdown | Kesimpulan akhir, keterbatasan, langkah lanjut |

### 15.3 Sumber yang Dirujuk Notebook

- Dataset RCAEval: <https://huggingface.co/datasets/phamquiluan/RCAEval>
- Implementasi resmi TS2Vec: <https://github.com/yuezhihan/ts2vec>
- Paket `RCAEval` (untuk RUN-native): `pip install "RCAEval[default]"`
