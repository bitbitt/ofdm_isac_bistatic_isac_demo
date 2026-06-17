# CONTEXT_TRANSFER — OFDM Bistatic ISAC (V2I comm + forward-scatter sensing)

> Dokumen handoff untuk melanjutkan proyek di chat baru. Berisi: ringkasan,
> lesson learned, status, lampiran yang harus dibawa, dan work plan.
> **File acuan utama: `ofdm_isac_bistatic_isac_demo.py`** (hasil merge final).

---

## 1. Ringkasan proyek (1 paragraf)

Sistem **ISAC bistatik** berbasis OFDM di SDR (USRP B210 sebagai TX "Lutetia",
LibreSDR B210-mini sebagai RX), fc = 5.9 GHz, FS = 40 MS/s. Satu waveform OFDM
dipakai serentak untuk **komunikasi V2I** (mengirim teks "STEI" via QPSK pada 8
subcarrier tengah + CRC16) dan **sensing deteksi objek** (forward-scattering /
gangguan LoS). Tujuan: membuktikan paradigma *integrated sensing and
communication* — satu sinyal, dua fungsi — sebagai langkah menuju 6G ISAC.

---

## 2. Konfigurasi teknis (locked)

| Parameter | Nilai |
|---|---|
| Hardware TX / RX | USRP B210 (serial 000000037) / LibreSDR (serial HQHGTFH) |
| fc | 5.9 GHz |
| FS winner | 40 MS/s (dari FS sweep Phase 1A) |
| Frame | STF + LTF1 + LTF2 + 26×DATA = 2320 sample, Nfft=64, CP=16 |
| Subcarrier | 46 data (8 comm QPSK + 38 sense BPSK) + 4 pilot |
| Paket V2I | text 32b + counter 8b + CRC16 = 56 bit |
| Gain | TX 90 dB, RX 76 dB |
| **Tuning JCAS final** | `--sense-threshold-k 3.5 --sense-min-score 1.2` |

---

## 3. Dua pendekatan sensing — yang dipakai vs dibuang

| | Range/echo (Phase 1C v1.4) | **Forward-scatter JCAS (Phase 1F)** |
|---|---|---|
| Prinsip | CIR → deteksi echo di delta-range | Gangguan amplitudo/fase LoS → MAD-CFAR |
| Status di ruangan kecil | **GAGAL** (artefak blind-zone) | **BERHASIL** (klaim deteksi utama) |
| Peran sekarang | KONTEKS visual saja | KLAIM DETEKSI |

**Kenapa range/echo gagal (penting, jangan diulang):** resolusi range δR ≈ 4.8 m,
blind zone ~5.6 m (skip 3 bin), sedangkan ruangan 4×4 m → delta-range target
forward-scatter < 2 m → **jatuh di dalam blind zone**, tak akan pernah jadi echo.
Yang terdeteksi (7.5/9.38 m) cuma residual clutter di tepi blind-zone, bukan
target. Detektor JCAS (gangguan LoS) tidak bergantung pada resolusi/blind-zone,
jadi itu alat yang benar untuk geometri ini.

---

## 4. PROOF (hasil tervalidasi — pakai ini untuk laporan)

Sumber angka resmi = run **tanpa plot**, tuning final (k=3.5, min_score=1.2),
panjang sama 1519 frame, post-warmup (frame > 40):

| Metrik | STATIC (diam) | WALK (gerak) | Pemisahan |
|---|---|---|---|
| Comm PRR | 100% | 100% | comm ✓ |
| Deteksi objek | **0.07%** (1 burst) | **2.50%** (6 burst) | ~37× |
| \|amp_hp\| 95-pct | 0.055 dB | 1.782 dB | **~32×** |
| Max JCAS score | 2.7 | 8.7 | — |

Tiga angka headline: **PRR 100%** (comm jalan), **kontras static-vs-walk 32×**
(sensing membedakan gerak/diam, bukan kebetulan), **false-alarm floor 0.07%**
(klaim kredibel). File proof: `static_k35.csv`, `walk_k35.csv`.

---

## 5. Lesson learned (paling mahal → murah)

1. **Right tool for the geometry.** Range/echo CIR salah-alat untuk forward-scatter
   ruangan kecil (target di blind-zone). Detektor gangguan-LoS yang benar.
   Sinyal target ADA di data (`habs_*`) — cuma dilihat di domain yang salah.
2. **Metrik amplitudo JCAS = `avg_amp`, BUKAN `max(cir_mag)`.** Inilah akar
   perbedaan akurasi antara mode. `max(cir_mag)` bisa "bin-hop" ke echo → menutupi
   shadowing LoS. `avg_amp` (broadband) stabil dan diskriminatif untuk shadowing.
3. **Perbandingan harus apple-to-apple.** Static vs walk wajib: tuning sama,
   panjang run sama, setting plot sama. Floor lama (0.31% @ k=4.5) tak valid
   dibanding walk @ k=3.5.
4. **Plot mengubah pengukuran.** matplotlib rebut CPU/GIL → OVF + jitter timing →
   false alarm static naik ~12× (0.07→0.82%). Pisahkan: **--plot untuk demo
   visual**, **tanpa --plot untuk angka resmi**.
5. **Buang warmup dari statistik.** ~40 frame pertama: history kosong → sigma
   kepentok lantai → skor meledak (pernah 3939) walau `object_detected=False`.
   Selalu hitung post-warmup.
6. **Diskriminator paling kokoh = `amp_hp`, bukan % deteksi.** % deteksi sensitif
   ke OVF; `amp_hp` 95-pct konsisten 32× di plot maupun no-plot.
7. **Jangan klaim artefak sebagai deteksi.** Echo blind-zone 7.5/9.38 m dipajang
   sebagai "konteks", bukan bukti — supaya kredibilitas tak jatuh saat ditanya.
8. **CFAR adaptif naik saat aktivitas** (thr 2.86→4.02). Itu sehat; event nyata
   tetap tembus (skor 13 ≫ thr 4).
9. **Arsitektur RX: single-thread + chunk recv besar** (RECV_CHUNK 65536) menang.
   Producer-consumer (v1.2) GAGAL: queue Python tak sanggup throughput chunk-kecil
   → perang GIL → jutaan drop.

---

## 6. ⚠️ Limitation yang masih terbuka — POSITIONING/GEOMETRI

Geometri bistatik sekarang **belum optimal**:
- Baseline TX–RX ~55 cm di ruangan 4×4 m → delta-path target sangat kecil.
- Akibatnya range-based sensing mati (sudah dijelaskan di §3). Forward-scatter
  JCAS "menyelamatkan", tapi geometri tidak dirancang untuk sensing yang kuat.
- Untuk forward-scattering yang benar, target idealnya **melintasi garis LoS
  TX–RX**; penempatan antena & baseline harus diatur agar lintasan target memotong
  LoS secara tegas. Ini sumber utama kenapa hasil masih "deteksi ada/tidak", belum
  ranging/velocity yang andal.

**Konsekuensi:** klaim saat ini terbatas pada **deteksi biner objek bergerak**
(ada/tidak). Ranging & velocity belum bisa diklaim (lihat work plan).

---

## 7. Yang HARUS dilampirkan di chat baru

Wajib:
1. `ofdm_isac_bistatic_isac_demo.py` — kode final (file acuan).
2. `static_k35.csv` + `walk_k35.csv` — proof resmi (no-plot, k=3.5).
3. `CONTEXT_TRANSFER.md` (file ini).

Opsional (kalau bahas demo/plot atau range):
4. `static_plot.csv` / `walk_plot.csv` — efek plot terhadap OVF/false-alarm.
5. Screenshot dashboard 6-panel saat dot "ADA OBJEK" nyala.

Kalimat pembuka chat baru yang disarankan:
> "Lanjutan proyek OFDM bistatic ISAC. Baca CONTEXT_TRANSFER.md dulu. File kode
> final = ofdm_isac_bistatic_isac_demo.py. Proof = static_k35.csv + walk_k35.csv.
> Saya mau lanjut ke [work plan item X]."

---

## 8. WORK PLAN (prioritas berikutnya)

| # | Tugas | Kenapa | Output |
|---|---|---|---|
| 1 | **Perbaiki geometri/positioning** (baseline TX–RX, antena agar target memotong LoS) | Akar limitation §6 | Setup terdokumentasi + run ulang static/walk |
| 2 | **ROC / Pd-vs-Pfa** (sweep `--sense-threshold-k`) dgn ground-truth lintasan | Klaim kuantitatif deteksi, bukan cuma kontras | Kurva ROC |
| 3 | **Ground-truth labeling** (catat kapan & berapa kali orang melintas) | Validasi jumlah burst vs lintasan nyata | CSV + label |
| 4 | **Turunkan OVF < 5% andal** (buffer/threading benar) | Syarat estimasi Doppler/velocity | OVF report |
| 5 | **Velocity/Doppler estimation** (setelah OVF beres) | Naik dari "ada/tidak" ke kecepatan | Doppler valid |
| 6 | **Range-Doppler Map (RDM)** — lihat varian `1.4-adv-gem` | Sensing 2D bila geometri & BW mendukung | RDM snapshot |
| 7 | **AoA / multi-target** (butuh array antena) | Lokalisasi, bukan cuma deteksi | — |
| 8 | **Penyelarasan 3GPP ISAC** (TR 22.837 / Rel-19→6G Rel-21) | Narasi standar untuk laporan/BRIN | Bagian dokumen |

---

## 9. Catatan operasional

- **CSV** hanya ditulis saat `--phase1b`. Auto-nama `isac_log_<epoch>.csv` bila
  `--log-csv` kosong. Biasakan kasih nama eksplisit bertanggal:
  `--log-csv run_$(date +%Y%m%d_%H%M).csv`.
- **Kalibrasi clutter** 300 frame pertama (ruangan diam) — itu untuk panel
  range/konteks. JCAS sendiri valid setelah ~40 frame warmup.
- **Nama file di disk** pakai titik & kurung (`ofdm_isac_bistatic_1.4-adv(c).py`)
  → wajib quote di shell. File final tanpa kurung, aman.
- Run resmi: **tanpa `--plot`**. Demo visual: **dengan `--plot`**.

---

*Konteks ini cukup untuk melanjutkan tanpa mengulang seluruh riwayat eksperimen.*
