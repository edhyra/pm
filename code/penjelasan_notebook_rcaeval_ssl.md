# Penjelasan Notebook: Self-Supervised Learning untuk Identifikasi Service Penyebab Gangguan pada RCAEval

> Dokumen ini menjelaskan isi dan cara kerja notebook `rcaeval_ssl_ts2vec_timemae_run.ipynb`.
> Penjelasan disusun berurutan mengikuti alur notebook, mulai dari tujuan, data, model, sampai cara evaluasi.
> Dokumen ini menjelaskan **rancangan dan implementasi**. Hasil eksperimen belum ada karena notebook belum dijalankan (lihat Bagian 11).

---

## Daftar Isi

1. [Latar Belakang dan Tujuan](#1-latar-belakang-dan-tujuan)
2. [Gambaran Umum Alur Kerja](#2-gambaran-umum-alur-kerja)
3. [Dataset dan Preprocessing](#3-dataset-dan-preprocessing)
4. [Pembagian Data (Split)](#4-pembagian-data-split)
5. [Windowing dan Representasi Case](#5-windowing-dan-representasi-case)
6. [Model yang Dibandingkan](#6-model-yang-dibandingkan)
7. [Protokol Evaluasi](#7-protokol-evaluasi)
8. [Konfigurasi Eksperimen](#8-konfigurasi-eksperimen)
9. [Alur Eksekusi Eksperimen](#9-alur-eksekusi-eksperimen)
10. [Penyajian Hasil dan Kesimpulan Otomatis](#10-penyajian-hasil-dan-kesimpulan-otomatis)
11. [Status Notebook Saat Ini](#11-status-notebook-saat-ini)
12. [Keterbatasan dan Catatan Metodologis](#12-keterbatasan-dan-catatan-metodologis)
13. [Rencana Pengembangan](#13-rencana-pengembangan)
14. [Lampiran](#14-lampiran)

---

## 1. Latar Belakang dan Tujuan

### 1.1 Masalah

Aplikasi modern biasanya dibangun dari banyak *microservice*. Ketika satu service bermasalah (misalnya CPU penuh, memori bocor, atau jaringan lambat), gangguan itu merambat ke service lain. Akibatnya banyak service terlihat bermasalah sekaligus. Menentukan service yang menjadi **sumber** gangguan disebut ***Root Cause Analysis* (RCA)**.

Setiap service menghasilkan banyak metrik berupa deret waktu (*time series*), misalnya penggunaan CPU, memori, dan latensi. RCA dapat dipandang sebagai masalah klasifikasi: *dari pola metrik, service mana yang menjadi akar masalah?*

### 1.2 Mengapa Self-Supervised Learning

Label akar masalah mahal karena harus ditentukan oleh insinyur yang menyelidiki insiden. Data metrik mentah tanpa label jauh lebih banyak. ***Self-Supervised Learning* (SSL)** memanfaatkan data tanpa label: model belajar mengenali struktur data lewat tugas buatan (*pretext task*), misalnya menebak bagian sinyal yang disembunyikan. Hasilnya adalah **representasi** (*embedding*) berupa vektor angka yang merangkum isi sebuah deret waktu. Representasi ini kemudian dipakai dengan sedikit label.

### 1.3 Pertanyaan yang Dijawab Notebook

1. Apakah representasi hasil SSL lebih baik daripada fitur statistik sederhana dan daripada encoder yang tidak dilatih (*random-init*)?
2. Apakah keunggulan SSL terlihat ketika label **sedikit** (*label efficiency*)?
3. Bagaimana SSL dibandingkan dengan model supervised yang dilatih dari nol dengan label?
4. Seberapa baik tiap model ketika diuji pada **jenis fault yang tidak pernah dilihat** saat training?

### 1.4 Tiga Keluarga Metode yang Dibandingkan

| Keluarga | Gagasan utama | Sumber implementasi |
|---|---|---|
| **TS2Vec** | *Contrastive learning* hierarkis pada deret waktu | Repo resmi `yuezhihan/ts2vec` |
| **TimeMAE** | *Masked autoencoder*: sebagian potongan sinyal disembunyikan, lalu model menebak representasinya | **Re-implementasi ringkas** di notebook (`TimeMAEMini`) |
| **RUN** | *Contrastive learning* tanpa pasangan negatif sebagai encoder, lalu *causal discovery* | **Adaptasi** di notebook (`ContrastiveNoNegMini`), hanya tahap encoder |

---

## 2. Gambaran Umum Alur Kerja

```
Data mentah (Hugging Face)
        │
        ▼
[1] Preprocessing  ── rapikan anomali, samakan kolom metrik, z-score
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
[8] Perbandingan dengan baseline + kesimpulan otomatis
```

Prinsip yang dijaga di seluruh alur: **semua model diperlakukan dengan prosedur yang sama** (preprocessing, split, dan probe identik). Dengan begitu perbedaan skor mencerminkan kualitas representasi, bukan perbedaan prosedur.

Seluruh alur di atas dijalankan **terpisah untuk tiap sistem RE1** (`RE1-OB`, `RE1-SS`, `RE1-TT`) dengan arsitektur dan hyperparameter yang sama. Hanya jumlah kanal input yang menyesuaikan diri dengan sistemnya. Hasil tiga sistem digabung di akhir untuk perbandingan lintas sistem.

---

## 3. Dataset dan Preprocessing

### 3.1 Dataset

Dataset yang dipakai adalah **RCAEval** (`phamquiluan/RCAEval` di Hugging Face). Setiap *case* adalah satu insiden berisi metrik deret waktu (disampling 1 detik), waktu penyuntikan fault (`inject_time`), dan label `root_cause_service` sebagai kelas.

Notebook hanya memakai generasi **RE1** dan menjalankan tiga sistemnya satu per satu: `RE1-OB` (Online Boutique), `RE1-SS` (Sock Shop), dan `RE1-TT` (Train Ticket). RE2 dan RE3 tidak dipakai. Tiap sistem diproses **terpisah** karena nama service berbeda di tiap sistem, sehingga kelas tidak bisa ditransfer lintas sistem. Arsitektur model dan seluruh hyperparameter dibuat sama untuk ketiganya.

Setiap sistem RE1 berisi 125 case: 5 service penyebab × 5 jenis fault (`cpu`, `delay`, `disk`, `loss`, `mem`) × 5 repetisi.

### 3.2 Masalah Data dan Penanganannya

| Masalah pada data | Penanganan di notebook | Fungsi |
|---|---|---|
| Jumlah metrik berbeda antar case (mis. 49–59 di RE1-OB) | Kolom disatukan (*union*). Kolom yang tidak ada di suatu case diisi **0** setelah normalisasi | `load_all` |
| `inject_time` rusak (mis. `re1ob_currencyservice_loss_1` bernilai 16.933.142) | Diganti **titik tengah seri**, karena RE1 menyuntik fault di tengah | `fix_inject` |
| Kolom `time` ganda pada sebagian case RE1-OB | Kolom duplikat dibuang | `read_case` |
| Kanal milik service yang **bukan kelas** di RE1 (mis. `frontend`, `emailservice`, `redis`) | Dibuang bila `PRUNE_CHANNELS=True`. Hanya kanal dengan prefix service yang termasuk kelas sistem itu yang dipakai | `load_all` |
| Case rusak: seri lebih pendek dari `WIN` atau tanpa periode faulty (mis. `re1ob_productcatalogservice_cpu_3`, 63 detik) | Case dibuang dan namanya dicetak | `load_all` |
| Nilai tak hingga atau kosong | `inf` diganti `NaN`, lalu diisi `ffill` dan `bfill` | `preprocess_case` |
| Skala metrik sangat berbeda (CPU, byte, milidetik) | **Z-score** per case | `preprocess_case` |

### 3.3 Normalisasi Z-score

Rumus per kolom metrik dalam satu case:

```
z = (x − mean_normal) / std_normal ,   lalu di-clip ke [−10, +10]
```

- `mean_normal` dan `std_normal` dihitung hanya dari **periode normal**, yaitu data sebelum `inject_time`.
- Dengan cara ini, nilai z yang besar berarti "menyimpang dari kondisi normal case itu sendiri". Inilah sinyal yang dicari.
- Jika `std_normal` hampir nol (metrik konstan), dipakai std seluruh seri. Jika masih nol, dipakai 1.
- *Clipping* ±10 mencegah satu lonjakan ekstrem mendominasi pelatihan.

### 3.4 Struktur Data Hasil Preprocessing

Hasil disimpan dalam dictionary global `DATA`:

| Kunci | Isi |
|---|---|
| `Z` | Daftar array `(T, C)` per case, yaitu deret waktu ternormalisasi |
| `inj` | Indeks baris tempat fault mulai |
| `cols` | Daftar kanal gabungan setelah kanal non-kelas dibuang (C kanal) |
| `y`, `classes` | Label service per case dan daftar kelas unik |
| `fault`, `rep` | Jenis fault dan nomor repetisi per case |
| `dropped`, `n_union` | Daftar service yang kanalnya dibuang, dan jumlah kanal gabungan sebelum dibuang |

`DATA` selalu berisi sistem yang **sedang diproses**. Fungsi `prepare_system(name)` memuat satu sistem, mencetak ringkasannya, dan menyimpannya ke `SYS_INFO[name]` (jumlah case, kanal, kelas, dan service yang dibuang) untuk dipakai sel hasil dan kesimpulan.
### 3.5 Data Sintetis untuk Uji Cepat (`DRY_RUN`)

Jika `DRY_RUN=True`, notebook membuat data buatan tanpa internet: 5 service × 4 jenis fault × 5 repetisi = 100 case, tiap service punya 6 kanal, panjang 721 detik. Fault disimulasikan sebagai gelombang sinus pada kanal service penyebab setelah titik tengah. Data ini sengaja memuat dua jebakan untuk menguji pipeline: satu case dengan `inject_time` rusak, dan case fault `delay` yang kehilangan satu kolom (menguji penyejajaran kolom). Tujuannya hanya memastikan kode berjalan (*smoke test*), **bukan** menghasilkan angka untuk dilaporkan.

### 3.6 Ringkasan Tiga Sistem RE1 setelah Pembersihan

| Sistem | Case | Kelas (5 per sistem) | Kanal gabungan | Kanal dipakai |
|---|---|---|---|---|
| RE1-OB | 124 (dari 125) | adservice, cartservice, checkoutservice, currencyservice, productcatalogservice | 86 | 39 |
| RE1-SS | 125 | carts, catalogue, orders, payment, user | 63 | 30 |
| RE1-TT | 125 | ts-auth-service, ts-order-service, ts-route-service, ts-train-service, ts-travel-service | 241 | 30 |

**Pengecekan kelas.** Kelas diambil dari `root_cause_service` pada indeks RE1, sehingga hanya 5 kelas per sistem yang muncul. Service seperti `frontend` hanya menjadi kelas di RE3-SS, sedangkan `emailservice` dan `recommendationservice` hanya di RE2/RE3. Jadi tidak ada label yang perlu dibuang. Yang dibuang adalah **kanal metrik** milik service yang tidak pernah menjadi kelas di RE1:

- **OB:** `frontend`, `frontend-check`, `frontend-external`, `emailservice`, `paymentservice`, `recommendationservice`, `shippingservice`, `redis`, `main`, `PassthroughCluster`.
- **SS:** `front-end`, `shipping`, `queue-master`, `rabbitmq`, `rabbitmq-exporter`, `session-db`, `istio-init`, dan `*-db`.
- **TT:** 59 service, antara lain `ts-ui-dashboard` dan semua `ts-*-mongo`.

Kanal pendamping seperti `carts-db` dan `ts-auth-mongo` ikut terbuang karena prefix-nya bukan kelas. Pencocokan memakai kesamaan penuh pada bagian nama sebelum `_` pertama.

---

## 4. Pembagian Data (Split)

### 4.1 Mengapa Split Acak Tidak Dipakai

Untuk tiap pasangan (service, jenis fault) ada **5 repetisi** yang hampir identik. Jika repetisi dibagi acak ke train dan test, model dapat "menghafal" case yang hampir sama dan skornya terlihat tinggi secara semu. Kondisi ini disebut **kebocoran data** (*data leakage*). Karena itu notebook memakai split berkelompok.

### 4.2 Tiga Skema Split (`make_splits`)

| Skema | Cara membagi | Yang diukur |
|---|---|---|
| **`easy`** | 5-fold stratified per service | Skenario paling longgar. Repetisi dari pasangan yang sama dapat muncul di train dan test |
| **`medium`** | Train = repetisi awal (1–3, sekitar 60%), test = repetisi akhir (4–5) | Generalisasi ke repetisi yang belum terlihat |
| **`hard`** | *Leave-one-fault-type-out*: train pada semua jenis fault **kecuali** satu, test pada jenis fault yang ditahan | Generalisasi ke **jenis fault baru**. Ini ukuran yang paling jujur |

Konfigurasi bawaan memakai `medium`.

### 4.3 Pretraining Tidak Boleh Melihat Data Test

Pretraining SSL memang tidak memakai label, tetapi tetap hanya memakai window dari **case train** pada split yang sedang berjalan. Hal ini mencegah informasi data test masuk lewat proses pretraining.

---

## 5. Windowing dan Representasi Case

### 5.1 Window

Model SSL menerima potongan deret waktu berukuran tetap (*window*) dengan bentuk `(N, WIN, C)`: N window, panjang `WIN = 128` detik, dan C kanal.

| Jenis window | Cara mengambil | Fungsi |
|---|---|---|
| **Pretraining** | Titik awal acak di seluruh seri case train (`pretrain_windows`) | Data untuk melatih SSL tanpa label |
| **Normal** | `k` window berjarak merata dari awal seri sampai sebelum `inject_time` | Gambaran kondisi sehat case ini |
| **Faulty** | `k` window berjarak merata mulai `inject_time` sampai akhir seri | Gambaran kondisi saat gangguan |

`k` adalah `EVAL_WIN`: 8 pada mode `QUICK`, 16 pada mode penuh.

### 5.2 Representasi Case

Satu case direpresentasikan oleh satu vektor:

```
R(case) = rata-rata embedding window faulty − rata-rata embedding window normal
```

Alasannya: yang berguna untuk RCA adalah **perubahan** akibat fault, bukan kondisi dasar sistem. Pengurangan ini menghilangkan ciri khas case yang tidak berkaitan dengan gangguan. Notebook menyediakan tiga mode lewat `CASE_REPR`:

| Mode | Isi vektor |
|---|---|
| `diff` (bawaan) | faulty − normal |
| `fault` | hanya faulty |
| `concat` | gabungan faulty dan (faulty − normal) |

---

## 6. Model yang Dibandingkan

Semua model SSL memakai antarmuka yang sama: `fit(windows)` untuk pretraining, lalu `encode(windows)` yang menghasilkan matriks `(n, d)`. Antarmuka seragam ini memudahkan pertukaran model di pipeline yang sama.

Ringkasan:

| Nama di `CFG["MODELS"]` | Jenis | Dilatih dengan label? | Pretraining? |
|---|---|---|---|
| `stat` | Baseline fitur statistik | Hanya probe | Tidak |
| `random_ts2vec` | Kontrol: TS2Vec tanpa pretraining | Hanya probe | Tidak |
| `ts2vec` | SSL contrastive hierarkis | Hanya probe | Ya |
| `timemae_mini` | SSL masked autoencoder | Hanya probe | Ya |
| `runcl_mini` | SSL contrastive tanpa negative pair | Hanya probe | Ya |
| `sup_conv` | Supervised dari nol | **Ya**, end-to-end | Tidak |
| RUN-native (opsional) | RUN asli dari paket RCAEval | Tidak | Tidak (dilatih per case) |

### 6.1 Baseline Statistik (`stat`)

Tanpa jaringan saraf. Untuk tiap case diambil bagian faulty (setelah `inject_time`), lalu dihitung per kanal:

- rata-rata z-score,
- nilai absolut maksimum z-score,
- standar deviasi z-score.

Hasilnya vektor berukuran 3 × C. Baseline ini menjawab: *apakah SSL lebih baik daripada ringkasan sederhana yang bisa dihitung langsung?*

### 6.2 TS2Vec dan Kontrol `random_ts2vec`

- **Cara kerja.** TS2Vec belajar secara *contrastive*: dua potongan yang berasal dari deret waktu yang sama harus punya representasi mirip, sedangkan potongan dari deret waktu lain harus berbeda. Kontras dilakukan secara **hierarkis** di berbagai skala waktu, baik antar-timestamp (temporal) maupun antar-instance.
- **Konfigurasi.** `output_dims=320`, `hidden_dims=64`, `depth=10`, `batch_size=16`. Embedding diambil dengan `encoding_window="full_series"`, sehingga satu window menghasilkan satu vektor.
- **Kontrol `random_ts2vec`.** Arsitektur yang sama persis, tetapi **tidak dilatih** (`epochs=0`). Jika `ts2vec` tidak lebih baik daripada `random_ts2vec`, berarti pretraining tidak menambah nilai. Arsitekturnya saja sudah cukup memberi ciri yang berguna. Kontrol ini penting agar manfaat SSL tidak diklaim berlebihan.

### 6.3 TimeMAE-mini (`TimeMAEMini`)

Re-implementasi ringkas bergaya TimeMAE. **Bukan** kode resmi TimeMAE.

**Gagasan.** Model diminta menebak bagian sinyal yang disembunyikan. Untuk menebaknya, model harus memahami pola waktu sinyal tersebut.

**Langkah kerja:**

1. **Patchify.** Window 128 detik dipotong menjadi 16 patch berukuran 8 detik (`PATCH=8`).
2. **Masking.** 60% patch (10 dari 16) disembunyikan secara acak, dan 6 patch tersisa terlihat.
3. **Encoder terdekoupel.** Encoder Transformer (dimensi 128, 3 lapis, 4 head) hanya melihat **patch yang tampak**. Token topeng tidak masuk encoder.
4. **Decoder.** Decoder Transformer 1 lapis menerima representasi patch tampak ditambah *mask token* di posisi yang disembunyikan, lalu memprediksi representasi posisi tersebut.
5. **Target dari encoder EMA.** Target prediksi dihasilkan oleh salinan encoder (*target encoder*) yang melihat **sinyal utuh**. Parameternya tidak dilatih lewat gradien, tetapi mengikuti encoder utama dengan *Exponential Moving Average* (`ema=0.99`).

**Dua fungsi loss:**

| Loss | Arti | Cara kerja |
|---|---|---|
| **MRR** (*Masked Representation Regression*) | Menebak nilai representasi | MSE antara prediksi dan representasi target di posisi yang disembunyikan |
| **MCC** (*Masked Codeword Classification*) | Menebak "kata kode" representasi | Representasi target dipetakan ke codeword terdekat (cosine) dalam codebook 128 kode. Model mengklasifikasi codeword itu dengan cross-entropy. Codebook diperbarui dengan EMA |

Loss total = `MRR + 0.5 × MCC`.

**Detail pelatihan:** AdamW (lr 5e-4, weight decay 1e-4), jadwal `OneCycleLR`, *gradient clipping* 1.0, dan EMA diperbarui setiap langkah.

**Encoding.** Setelah pelatihan, encoder utama melihat semua patch dan hasilnya dirata-ratakan sepanjang waktu menjadi satu vektor 128 dimensi per window.

### 6.4 RUN-adapted (`ContrastiveNoNegMini`)

**Latar belakang.** RUN asli memakai *contrastive learning* tanpa pasangan negatif untuk memperkaya encoder, lalu melanjutkan dengan *neural Granger causal discovery* dan *PageRank* untuk mengurutkan penyebab. Notebook ini hanya mengambil **tahap encoder** supaya bisa dibandingkan dengan TS2Vec dan TimeMAE dalam protokol linear probe yang sama. Tahap kausal tidak ada di sini. Augmentasi dan arsitektur adalah pilihan penulis notebook, bukan kode asli RUN.

**Cara kerja (gaya SimSiam):**

1. Dari satu batch window dibuat **dua view** dengan augmentasi berbeda:
   - *crop* acak 75% dari panjang window,
   - perkalian skala acak per kanal,
   - *jitter* (derau Gaussian kecil),
   - *time masking* (15% titik waktu dinolkan),
   - *channel dropout* (10% kanal dinolkan).
2. Kedua view melewati encoder `f`, lalu *projector* `g`. Salah satu hasilnya dilewatkan lagi ke *predictor* `h`.
3. Loss adalah **negatif cosine similarity** simetris antara prediksi satu view dan proyeksi view lainnya.
4. ***Stop-gradient*** dipasang pada sisi target (`z.detach()`). Tanpa pasangan negatif, mekanisme ini bersama predictor mencegah model *collapse*, yaitu menghasilkan vektor yang sama untuk semua input.

**Encoder `_ConvEnc`:** konvolusi 1D dengan 4 blok residual berdilasi (dilasi 1, 2, 4, 8; kernel 5; BatchNorm; GELU). Keluarannya diagregasi dengan *mean pooling* dan *max pooling* sepanjang waktu, lalu diproyeksikan ke 128 dimensi. Saat encoding hanya encoder `f` yang dipakai, tanpa projector dan predictor.

### 6.5 Pembanding Supervised (`sup_conv`)

- Memakai encoder `_ConvEnc` yang **sama** dengan RUN-adapted, ditambah satu lapisan linear untuk klasifikasi.
- Dilatih **dengan label** end-to-end (cross-entropy, derau Gaussian 0,05 sebagai augmentasi, weight decay 1e-2) pada window faulty.
- Prediksi satu case = rata-rata probabilitas softmax dari window-window faulty case tersebut.
- Dilatih ulang untuk **setiap** pecahan label. Berbeda dengan model SSL, tidak ada pretraining yang dipakai ulang.
- Perannya adalah menjawab: *jika label sedikit, apakah SSL benar-benar lebih baik daripada langsung melatih dengan label?*

### 6.6 RUN-native (Opsional)

Bagian 8 notebook menjalankan RUN asli dari paket `RCAEval`. Perbedaannya dengan model lain:

| Aspek | Linear probe (model SSL) | RUN-native |
|---|---|---|
| Label | Dipakai untuk probe | **Tidak dipakai sama sekali** |
| Keluaran | Kelas service | **Ranking metrik**, dipetakan ke service lewat prefix nama metrik |
| Pelatihan | Satu kali untuk semua case | **Per case** (lambat) |
| Metrik | accuracy, macro-F1, Top-3 | Top-1, Top-3 |

Setelan yang dipilih penulis notebook: jendela waktu ±600 detik di sekitar `inject_time` dan maksimal 10 case uji (`RUN_MAX_CASES`). Fitur ini mati secara bawaan (`RUN_NATIVE=False`) dan perlu `pip install "RCAEval[default]"`. Angkanya **tidak boleh dibandingkan langsung** dengan linear probe karena jenis keluarannya berbeda.

---

## 7. Protokol Evaluasi

### 7.1 Linear Probe

Semua representasi case dievaluasi dengan pengklasifikasi yang sama:

```
StandardScaler  →  LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
```

**Linear probe** berarti encoder dibekukan dan hanya pengklasifikasi linear di atasnya yang dilatih. Karena pengklasifikasinya sederhana, skor yang baik hanya mungkin jika representasinya sendiri sudah memisahkan kelas dengan baik. `class_weight="balanced"` menyeimbangkan kelas yang jumlah case-nya tidak sama.

### 7.2 Metrik

| Metrik | Arti |
|---|---|
| **Accuracy** | Persentase case yang service-nya ditebak benar |
| **Macro-F1** | Rata-rata F1 per kelas, tiap kelas berbobot sama, sehingga tidak terdominasi kelas besar |
| **Top-3 accuracy** | Persentase case yang service sebenarnya ada di tiga prediksi teratas. Relevan untuk RCA karena insinyur sering memeriksa beberapa kandidat teratas |

Akurasi acak (*chance*) dihitung `1 / jumlah kelas` dan digambar sebagai garis pembanding pada grafik.

### 7.3 Label Efficiency

Probe dilatih dengan sebagian label train: `LABEL_FRACS = [0.1, 0.2, 0.5, 1.0]` (10%, 20%, 50%, 100%).

- Pengambilan sampel dilakukan **per kelas** dan minimal 1 case per kelas (`subsample_train`), sehingga semua kelas tetap terwakili.
- Pecahan 1% tidak dipakai karena jumlah case terlalu kecil.
- Kedua ujung punya arti berbeda:
  - **Label penuh (100%)** mengukur **kualitas representasi**.
  - **Label sedikit (10%)** mengukur **manfaat SSL**.

### 7.4 Akurasi per Jenis Fault

Pada label penuh, notebook mencatat apakah setiap case uji ditebak benar, lalu merata-ratakannya per (model, jenis fault). Dengan begitu terlihat jenis fault mana yang sulit dikenali.

---

## 8. Konfigurasi Eksperimen

Seluruh pengaturan ada dalam dictionary `CFG`.

### 8.1 Parameter Utama

| Parameter | Nilai bawaan | Arti |
|---|---|---|
| `SYSTEMS` | `RE1-OB`, `RE1-SS`, `RE1-TT` | Sistem yang dijalankan satu per satu. Semua harus berawalan `RE1-` |
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
| `PRUNE_CHANNELS` | `True` | Buang kanal milik service yang bukan kelas |

Hyperparameter model (dimensi, jumlah lapis, learning rate, dan seterusnya) satu set untuk semua sistem. Yang berbeda antar sistem hanya jumlah kanal input.
### 8.2 Mode `QUICK` dan Mode Penuh

| Parameter | `QUICK=True` (bawaan) | `QUICK=False` |
|---|---|---|
| Seed (`SEEDS`) | 1 (`[0]`) | 3 (`[0, 1, 2]`) |
| Window pretraining per case (`PRE_WIN`) | 24 | 80 |
| Window evaluasi per case (`EVAL_WIN`) | 8 | 16 |
| Epoch TS2Vec | 3 | 20 |
| Epoch TimeMAE-mini | 15 | 60 |
| Epoch RUN-adapted | 15 | 60 |
| Epoch supervised | 15 | 60 |

Mode `QUICK` dimaksudkan hanya untuk memastikan pipeline berjalan. Hasil yang akan dilaporkan sebaiknya memakai mode penuh dengan minimal 3 seed.

### 8.3 Reprodusibilitas

Fungsi `set_seed` mengatur seed `random`, `numpy`, dan `torch` (termasuk CUDA) sebelum membangun dan melatih tiap model.

---

## 9. Alur Eksekusi Eksperimen

Fungsi `run_all_systems` memanggil `prepare_system` lalu `run_experiments` untuk tiap sistem, kemudian menggabungkan hasilnya dengan kolom `system`. Perulangan lengkapnya:

```
untuk setiap sistem di CFG["SYSTEMS"]:               # run_all_systems
  muat data sistem, buang kanal non-kelas            # prepare_system
  untuk setiap seed:                                  # run_experiments
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

Catatan penting tentang alur ini:

- **Data pretraining dipakai bersama** oleh semua model SSL dalam satu split, sehingga perbandingan adil.
- **Penanganan kegagalan.** Jika satu model gagal (misalnya repo TS2Vec belum di-clone), notebook mencetak `[GAGAL]` dan melanjutkan ke model berikutnya, tidak menghentikan seluruh eksperimen.
- **Setiap sistem dilatih dari awal.** Tidak ada bobot yang dibawa antar sistem. Waktu jalan kira-kira tiga kali lipat dibanding satu sistem.
- **Hasil** dikumpulkan dalam dua DataFrame: `results` (sistem, model, split, seed, pecahan label, jumlah case train, dan metrik) serta `per_fault` (sistem, benar atau salah per case per jenis fault).

---

## 10. Penyajian Hasil dan Kesimpulan Otomatis

### 10.1 Tabel dan Grafik (Bagian 7 Notebook)

Untuk **setiap sistem** (judul grafik memuat nama sistem):

- Tabel rata-rata dan simpangan baku (mean ± std antar seed) untuk accuracy, macro-F1, dan Top-3 per (model, pecahan label).
- **Grafik kiri:** kurva *label efficiency*, yaitu accuracy terhadap pecahan label dengan error bar dan garis *chance* (`1 / jumlah kelas` sistem itu).
- **Grafik kanan:** diagram batang ketiga metrik pada label penuh.
- Tabel akurasi per jenis fault.

Setelah itu ada **tabel lintas sistem**: accuracy dan macro-F1 pada label penuh dengan baris = model dan kolom = sistem (`RE1-OB`, `RE1-SS`, `RE1-TT`).

### 10.2 Kesimpulan Otomatis (`auto_conclusion`)

Agar tidak ada angka yang ditulis manual dan kesimpulan selalu sesuai hasil terbaru, notebook menyusun kesimpulan dari data hasil. Fungsi `auto_conclusion(results, system, run_native_df)` dipanggil **sekali per sistem**, dengan isi sebagai berikut:

1. Model terbaik pada label penuh dan pada label paling sedikit.
2. Perbandingan setiap model SSL dengan `random_ts2vec` dan `stat` pada label paling sedikit dan paling banyak. Selisih rata-rata `d` diberi vonis:
   - **lebih baik** bila `d` positif dan lebih besar daripada std terbesar kedua model,
   - **lebih buruk** bila `d` negatif dan nilai mutlaknya melebihi std itu,
   - **tidak beda jelas** untuk kondisi lainnya.
3. Selisih tiap model SSL terhadap `sup_conv` pada label sedikit.
4. Hasil RUN-native, bila dijalankan.
5. Peringatan otomatis jika jumlah seed kurang dari 3.

### 10.3 Kriteria "SSL Bermanfaat"

Notebook mendefinisikan pretraining bermanfaat **hanya jika ketiganya terpenuhi**:

1. mengalahkan encoder random-init dengan arsitektur sama (`random_ts2vec`),
2. mengalahkan baseline fitur statistik (`stat`),
3. keunggulannya terlihat pada label sedikit.

---

## 11. Status Notebook Saat Ini

| Aspek | Keadaan |
|---|---|
| Jumlah sel | 25 (13 markdown, 12 code) |
| Output tersimpan | **Tidak ada**. Semua sel code belum memiliki output atau nomor eksekusi |
| Hasil eksperimen | **Belum ada**. Tidak ada angka accuracy, F1, atau grafik yang bisa dikutip |
| Kesimpulan tertulis (markdown) | Berisi kerangka metodologis, cara membaca hasil, keterbatasan, dan langkah lanjut. **Tanpa angka** |
| Konfigurasi aktif | `SYSTEMS` = `RE1-OB`, `RE1-SS`, `RE1-TT`, `PRUNE_CHANNELS=True`, `QUICK=True`, `SPLIT="medium"`, 1 seed, `DRY_RUN=False`, `RUN_NATIVE=False` |
| Pengujian alur | Data sintetis dengan keenam model, dan data asli ketiga sistem RE1 dengan `stat`, `random_ts2vec`, `sup_conv` (`QUICK=True`, 1 seed). Model `ts2vec`, `timemae_mini`, dan `runcl_mini` belum diuji pada data asli. Angka dari uji ini hanya memastikan kode berjalan dan tidak dicatat sebagai hasil |

Konsekuensi untuk laporan: bagian rancangan dan metode (Bagian 3–9) sudah dapat ditulis dari kode. Bagian hasil dan pembahasan baru dapat ditulis setelah notebook dijalankan. Sebaiknya jalankan dulu satu sistem dengan `QUICK=True` untuk memastikan semua model berjalan, lalu ketiga sistem dengan `QUICK=False` dan minimal 3 seed.

---

## 12. Keterbatasan dan Catatan Metodologis

### 12.1 Keterbatasan yang Dinyatakan di Notebook

- **Data kecil.** Hanya puluhan case per kelas, dengan repetisi yang hampir duplikat, sehingga variansi antar split besar. Disarankan minimal 3 seed.
- **Bukan implementasi resmi.** TimeMAE-mini dan RUN-adapted adalah re-implementasi atau adaptasi. Hasilnya tidak mewakili TimeMAE atau RUN resmi. Untuk klaim publikasi perlu diulang dengan repo resmi.
- **Tanpa transfer lintas sistem.** Kelas adalah service spesifik per sistem. Tiap sistem RE1 punya 5 kelas dan dijalankan terpisah, jadi angka antar sistem tidak ekuivalen (kelas dan jumlah kanalnya berbeda).
- **Kanal non-kelas dibuang** (`PRUNE_CHANNELS`), termasuk kanal DB pendamping. Pengaruhnya terhadap hasil belum diuji.
- **Pilihan preprocessing memengaruhi hasil**, yaitu normalisasi dengan periode normal, window 128 detik, clip ±10, dan pengisian kolom hilang dengan 0. Hal ini perlu diuji lewat ablation.

### 12.2 Catatan Tambahan dari Pembacaan Kode

Catatan berikut berasal dari membaca kode (bukan dari hasil eksekusi) dan layak diperhatikan saat menafsirkan hasil:

1. **Std = 0 pada 1 seed.** Dengan satu seed, simpangan baku antar seed bernilai 0 (kode mengisi `NaN` dengan 0). Ambang "1 std" pada kesimpulan otomatis lalu praktis menjadi "selisih positif apa pun". Karena itu vonis "lebih baik" atau "lebih buruk" pada mode `QUICK` tidak bermakna. Notebook memang menambahkan peringatan bila seed kurang dari 3.
2. **Pembanding supervised tidak persis setara.** `sup_conv` dilatih pada window faulty saja, sedangkan model SSL memakai representasi selisih faulty − normal. Jadi perbandingan keduanya bukan *apple-to-apple* sepenuhnya.
3. **Split `easy` paling optimistis.** Repetisi hampir duplikat dapat jatuh di train dan test sekaligus. Skor `easy` sebaiknya dibaca sebagai batas atas, bukan ukuran generalisasi.
4. **Normal vs faulty bergantung `inject_time`.** Window normal dan faulty ditentukan dari titik injeksi. `fix_inject` hanya mengoreksi nilai yang berada di luar rentang waktu seri. Jika `inject_time` salah tetapi masih di dalam rentang, nilai itu tidak terdeteksi dan representasi selisih ikut tercemar.
5. **Kolom hilang diisi 0.** Kanal yang tidak dimiliki suatu case terlihat sebagai "tidak ada penyimpangan", yang bisa saja bukan kondisi sebenarnya.
6. **Penamaan kanal RE1-OB tidak seragam.** Dari 39 kanal yang dipakai di OB, hanya 10 yang ada di semua case. Ada dua skema nama: 50 case memakai `*_latency` dan `*_load`, sedangkan 75 case memakai `*_latency-50`, `*_latency-90`, dan `*_workload`. Kolom `*_error` hanya ada di sebagian kecil case (mis. `cartservice_error` hanya di 6 dari 125 case). Karena kolom yang hilang diisi 0, tiap case OB memiliki banyak kanal kosong dan dua skema nama dianggap kanal yang berbeda. SS dan TT jauh lebih rapi (26 dan 25 dari 30 kanal ada di semua case). Hal ini sudah ada sebelum kanal non-kelas dibuang dan belum ditangani.
7. **Pembuangan kanal memakai himpunan kelas.** Pemilihan input memanfaatkan fakta bahwa penyebabnya pasti salah satu dari 5 service. Ini wajar untuk klasifikasi 5 kelas, tetapi berbeda dari RCA nyata saat semua service menjadi kandidat. Kanal gejala seperti `frontend` yang mungkin informatif juga ikut hilang.
8. **Satu case dibuang di RE1-OB.** `re1ob_productcatalogservice_cpu_3` hanya 63 detik dan tidak punya periode faulty, sehingga `productcatalogservice` dengan fault `cpu` tinggal 4 repetisi. Langkah ini ditambahkan di luar keputusan kelompok dan dapat dimatikan dengan menghapus blok filter di `load_all`.

---

## 13. Rencana Pengembangan

Langkah lanjut dari notebook:

1. Jalankan `QUICK=False` dengan minimal 3 seed untuk ketiga split (`easy`, `medium`, `hard`).
2. Ablation pembuangan kanal (`PRUNE_CHANNELS=True` vs `False`) dan penyeragaman nama kanal RE1-OB (`latency` dengan `latency-50`, `load` dengan `workload`). RE2 dan RE3 di luar cakupan.
3. Lakukan ablation: panjang window, `CASE_REPR` (`diff` vs `fault` vs `concat`), mask ratio, dan ukuran patch.
4. Tambahkan pipeline B (embedding per service lalu ranking) agar sebanding dengan metrik RCA standar (Avg@k).
5. Ganti TimeMAE-mini dengan repo resmi dan RUN-adapted dengan RUN penuh sebagai konfirmasi.

---

## 14. Lampiran

### 14.1 Glosarium

| Istilah | Penjelasan singkat |
|---|---|
| **RCA** (*Root Cause Analysis*) | Menentukan sumber awal suatu gangguan |
| **Microservice** | Aplikasi yang dipecah menjadi banyak layanan kecil yang saling memanggil |
| **Deret waktu** (*time series*) | Data yang diukur berurutan menurut waktu, misalnya CPU tiap detik |
| **SSL** | Belajar dari data tanpa label lewat tugas buatan |
| **Embedding / representasi** | Vektor angka yang merangkum isi suatu data |
| **Pretext task** | Tugas buatan yang dipakai SSL, misalnya menebak bagian yang disembunyikan |
| **Contrastive learning** | Melatih model agar data yang "mirip" memiliki embedding dekat dan yang berbeda berjauhan |
| **Masked autoencoder** | Menyembunyikan sebagian input lalu meminta model memulihkannya |
| **EMA** (*Exponential Moving Average*) | Rata-rata bergerak yang memberi bobot besar pada nilai lama, sehingga perubahannya halus |
| **Stop-gradient** | Menahan aliran gradien pada satu cabang jaringan |
| **Collapse** | Kegagalan SSL ketika model menghasilkan embedding yang sama untuk semua input |
| **Linear probe** | Menguji embedding dengan pengklasifikasi linear di atas encoder yang dibekukan |
| **Label efficiency** | Seberapa baik model bekerja ketika label yang tersedia sedikit |
| **Data leakage** | Informasi data uji "bocor" ke data latih, membuat skor tampak lebih tinggi dari kenyataan |
| **Z-score** | Nilai yang dinyatakan dalam satuan simpangan baku dari rata-rata |
| **Ablation** | Mengubah atau menghapus satu komponen untuk melihat pengaruhnya |
| **Top-k accuracy** | Benar jika jawaban sebenarnya ada di k prediksi teratas |

### 14.2 Peta Sel Notebook

| Sel | Jenis | Isi |
|---|---|---|
| 0 | Markdown | Judul, alur, catatan kejujuran implementasi |
| 1–2 | Markdown, Code | Setup: install paket, clone repo TS2Vec |
| 3 | Code | Import, `CFG`, `set_seed` |
| 4–5 | Markdown, Code | Data: loader, preprocessing, data sintetis, `load_all`, `prepare_system` |
| 6–7 | Markdown, Code | Split anti-bocor, windowing |
| 8–9 | Markdown, Code | Linear probe, metrik, baseline statistik, `embed_all` |
| 10–11 | Markdown, Code | TS2Vec dan kontrol random |
| 12–13 | Markdown, Code | TimeMAE-mini |
| 14–15 | Markdown, Code | RUN-adapted dan `sup_conv` |
| 16–17 | Markdown, Code | `build_embedder`, `run_experiments`, `run_all_systems` |
| 18–19 | Markdown, Code | Tabel dan grafik hasil per sistem, tabel lintas sistem |
| 20–21 | Markdown, Code | RUN-native (opsional, per sistem) |
| 22–23 | Markdown, Code | Kesimpulan otomatis per sistem |
| 24 | Markdown | Kesimpulan akhir, keterbatasan, langkah lanjut |

### 14.3 Sumber yang Dirujuk Notebook

- Dataset RCAEval: <https://huggingface.co/datasets/phamquiluan/RCAEval>
- Implementasi resmi TS2Vec: <https://github.com/yuezhihan/ts2vec>
- Paket `RCAEval` (untuk RUN-native), dipasang dengan `pip install "RCAEval[default]"`.
