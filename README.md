# Kernel-PAO

**Kernel-Space Posterior-Probability Adaptive Oversampling** for imbalanced rolling-bearing health monitoring.

This repository contains the implementation, preprocessed feature subset, results, and figures of the revised manuscript
*"A Posterior Probability-Guided Kernel-Space Oversampling ML Method for Imbalanced Rolling Bearing Health Monitoring"* (submitted to JEAS).

> The method was formerly named **Kernel-CCP**; it has been renamed **Kernel-PAO** throughout the revised manuscript and code.

---

## Dataset

We use the **UORED-VAFCLS** dataset (University of Ottawa Rolling-element Dataset — Vibration and Acoustic Faults under Constant Load and Speed Conditions).

- Authors: Mert Sehri, Patrick Dumond (2023)
- Publisher: Mendeley Data
- DOI: `10.17632/y2px5tg92h.5`
- URL: <https://data.mendeley.com/datasets/y2px5tg92h/5>
- Format: `.csv` / `.xlsx` / `.mat` (raw vibration, 42 kHz, column `Accelerometer`)

> The raw dataset is publicly available and is **not** re-hosted here. This repository only ships the derived feature subset and reproduction code.

---

## Repository structure

```
.
├── code/v2/                         # Modular implementation (strict protocol)
│   ├── config.py                    # Paths and fixed hyper-parameters
│   ├── feature_extraction.py        # Non-overlapping windowing + statistical features
│   ├── data_utils.py                # Grouped CV + imbalanced-subset construction
│   ├── kernel_pao.py                # Core Kernel-PAO oversampler
│   ├── baselines.py                 # SMOTE / ASVB-SMOTE / under- & over-sampling baselines
│   ├── metrics_utils.py             # G-mean, F-value, AUC, PR-AUC, recall, precision
│   ├── experiments.py               # Main comparisons, kernel/N ablation, noise study
│   ├── stats_tests.py               # Dietterich 5×2 cross-validation paired t-test
│   ├── multiseed.py                 # Multi-seed reproducibility (seeds 42 / 0 / 2024)
│   ├── plot_figures.py              # Regenerates figures_v2/
│   └── run_all.py                   # One-command reproduction
├── data/                            # Preprocessed feature subset (DO NOT include raw data)
│   └── UORED_features_nonoverlap.csv
├── results_v2/                      # Strict-protocol results (all tables in the manuscript)
├── figures_v2/                      # Figure 1–8 (PNG + PDF)
├── requirements.txt
└── README.md
```

---

## Reproducibility

```bash
pip install -r requirements.txt
cd code/v2
python run_all.py      # reproduces results_v2/ and figures_v2/
```

**Strict evaluation protocol** (this is what the revised manuscript reports):

- **Non-overlapping windows**: `window = step = 5000` (no overlap).
- **Grouped cross-validation**: `StratifiedGroupKFold` with `source_file` as the group variable, so all windows from the same raw vibration file stay within a single fold (no leakage).
- **Class direction**: `faulty = minority (y = 1)`, `healthy = majority (y = 0)`; the faulty class is under-sampled to build IR = 1:2 / 1:4 / 1:8 / 1:16.
- **Nested hyper-parameter selection**: outer grouped 5-fold for evaluation only; inner 3-fold selects kernel / N / bandwidth σ; the test fold never participates in any selection.
- **Statistical testing**: Dietterich 5×2 cross-validation paired *t*-test (df = 5).
- **Seeds**: primary `random_state = 42`; `multiseed.py` additionally runs seeds 0 and 2024.

---

## Protocol correction relative to the original submission

The originally reported numbers were inflated by three issues, all now fixed:

1. **Data leakage** — 50%-overlapped windows + random (non-grouped) k-fold split shared windows across train/test.
2. **Class-direction reversal** — healthy was wrongly treated as the minority class.
3. **Test-set tuning** — kernel / N were selected on test-fold performance.

Under the corrected protocol, Kernel-PAO performs **on par with** the best oversampling baselines (SMOTE, uniform) — inter-method differences are **not statistically significant** — which the revised manuscript reports honestly.