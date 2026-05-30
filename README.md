#    Statistical Audit: pandas-dev/pandas
**STI 2025 — Group 13 Final Assignment**

## Research Questions

| # | Pertanyaan | Teknik | Notebook |
|---|-----------|--------|----------|
| **RQ1** | "Berapa probabilitas sebuah Pull Request di pandas mendapatkan review dalam 7 hari pertama, dan bagaimana distribusi posterior-nya?" | MLE Bernoulli + Beta Posterior | `02_estimation.ipynb`, `03_confidence_interval.ipynb` |
| **RQ2** | "Apakah rata-rata jumlah issue yang dibuka per minggu berbeda secara signifikan antara sebelum dan sesudah rilis pandas 2.0 (April 2023)?" | MLE Poisson + Two-sample Z-test | `02_estimation.ipynb`, `04_hypothesis_testing.ipynb` |
| **RQ3** | "Berapa probabilitas sebuah issue di pandas membutuhkan lebih dari 30 hari untuk ditutup, dan bagaimana distribusinya?" | Monte Carlo Simulation | `05_simulation.ipynb` |

---

## Struktur Repository

```
stat-audit-pandas-sti-2025/
├── README.md                
├── AI_USAGE_LOG.md                   ← log penggunaan AI
├── data/
│   ├── raw/            ← original data
│   │   ├── issues_raw.csv            
│   │   └── prs_raw.csv              
│   └── clean/            ← clean data
│       ├── issues_clean.csv          
│       └── prs_clean.csv           
├── src/
│   ├── estimator.py                  ← Member B: MLE & Beta posterior
│   ├── inference.py                  ← Member C: CI & credible interval
│   ├── hypothesis.py                 ← Member D: Z-test
│   └── simulation.py                 ← Member E: Monte Carlo, Bloom Filter, MCMC
├── notebooks/
│   ├── 01_eda.ipynb                  ← Member A: EDA
│   ├── 02_estimation.ipynb           ← Member B: Estimasi parameter
│   ├── 03_confidence_interval.ipynb  ← Member C: Interval kepercayaan
│   ├── 04_hypothesis_testing.ipynb   ← Member D: Uji hipotesis
│   └── 05_simulation.ipynb           ← Member E: Simulasi
├── report/
│   └── statistical_health_report.pdf
├── presentation/
│   └── video_link.md
└── requirements.txt
```

---

## Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/rafaelinm/stat-audit-pandas-sti-2025.git
cd stat-audit-pandas-sti-2025
pip install -r requirements.txt
```

### 2. Jalankan Notebook Secara Berurutan

```bash
jupyter notebook
```

Urutan wajib: `01_eda` → `02_estimation` → `03_confidence_interval` & `04_hypothesis_testing` → `05_simulation`


---

## Temuan Utama

| Layer | Temuan | Implikasi |
|-------|--------|-----------|
| **Estimasi (B)** | θ̂ merge rate = 0.6677 (66,77%); λ̂ issues/minggu = 10,65; Beta posterior α=2004, β=998 | Probabilitas PR baru di-merge adalah sekitar 66,77%. Dengan jumlah data sebesar 3.000 PR, estimasi sudah sangat presisi — distribusi posterior sangat sempit di sekitar nilai MLE |
| **CI (C)** | 95% CI merge rate = (*TBD*, *TBD*) | *TBD* |
| **Hipotesis (D)** | p-value = *TBD* | *TBD* |
| **Simulasi (E)** | P(issue > 30 hari) ≈ *TBD* | *TBD* |

---

## Tim

| Member | Nama | Peran | Layer |
|--------|------|-------|-------|
| A & B | Rafaeli Niamonio Marundrury | Data Engineer & Estimation Analyst | EDA, pengumpulan & pembersihan data, MLE, Beta posterior, visualisasi likelihood |
| C | [Nama] | Inference Analyst | Confidence interval, credible interval |
| D | [Nama] | Hypothesis Analyst | Z-test, interpretasi p-value |
| E | [Nama] | Computation Analyst | Monte Carlo, Bloom Filter, MCMC |

---

## Sumber Data

| Atribut | Detail |
|---------|--------|
| **Repositori** | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) |
| **Endpoint** | GitHub REST API v3 (`/issues`, `/pulls`) |
| **Rentang data** | Issues: 29 September 2010 — 19 Mei 2026; PR: 19 Juni 2025 — 29 Mei 2026 |
| **Tanggal akses** | 30 Mei 2026 |
| **Total issues** | 2.776 issues (setelah pembersihan) |
| **Total PR** | 3.000 PR (2.003 merged, 997 ditolak) |
| **Titik potong analisis** | Rilis pandas 2.0 — 3 April 2023 |

**Keterbatasan yang diketahui:**
- GitHub API membatasi 5.000 request/jam per token; data sangat lama mungkin tidak terkumpul seluruhnya
- Kolom `changed_files`, `additions`, `deletions` pada PR hanya tersedia via endpoint detail (tidak diambil untuk efisiensi)
- Issue yang dikonversi dari PR (atau sebaliknya) mungkin terhitung ganda jika tidak difilter dengan benar

---

## Referensi

- Tsun, A. (2020). *Introduction to Probability and Statistics*. — Referensi utama seluruh formula
- GitHub REST API Documentation: https://docs.github.com/en/rest
- pandas release history: https://pandas.pydata.org/docs/whatsnew/index.html

---

## Penggunaan AI

Penggunaan AI tools didokumentasikan lengkap di [`AI_USAGE_LOG.md`](./AI_USAGE_LOG.md).