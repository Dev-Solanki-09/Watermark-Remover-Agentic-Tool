# wmclean Benchmark & Baseline Comparison Table

Generated: 2026-10-22 | Quick Suite: True | Total Tests: 18

### Metric Definitions
- **Paper Residual (MAE)**: Mean absolute pixel intensity error on background paper within the watermark zone (0.0 is perfect).
- **Ink Loss Rate (%)**: Fraction of ground-truth foreground text and diagram strokes eroded or lost (0.00% is perfect).
- **Outside Mask Change (%)**: Fraction of pixels modified outside the ground-truth watermark mask (0.00% is perfect).
- **Time (s)**: Wall-clock execution time per page.

### Results Table

| Fixture ID | Method | Paper Residual (MAE) | Ink Loss Rate (%) | Outside Mask Change (%) | Time (s) | RAM (MB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `BM_01_diag_gray_150` | **No-Op (Identity)** | 32.98 | 0.04% | 0.00% | 0.05s | 0 |
| `BM_01_diag_gray_150` | **Oracle Threshold** | 0.11 | 1.97% | 0.44% | 5.05s | 0 |
| `BM_01_diag_gray_150` | **wmclean (active)** | 32.98 | 70.06% | 2.05% | 25.85s | 347 |
| `BM_02_subtle_200` | **No-Op (Identity)** | 15.32 | 0.00% | 0.00% | 0.01s | 0 |
| `BM_02_subtle_200` | **Oracle Threshold** | 0.01 | 0.00% | 0.19% | 7.82s | 0 |
| `BM_02_subtle_200` | **wmclean (active)** | 15.81 | 52.91% | 1.57% | 132.92s | 272 |
| `BM_03_heavy_dark_200` | **No-Op (Identity)** | 82.05 | 0.09% | 0.00% | 0.01s | 0 |
| `BM_03_heavy_dark_200` | **Oracle Threshold** | 79.35 | 0.46% | 0.28% | 7.59s | 0 |
| `BM_03_heavy_dark_200` | **wmclean (active)** | 80.93 | 53.42% | 1.49% | 59.08s | 137 |
| `BM_04_red_stamp_200` | **No-Op (Identity)** | 54.74 | 0.01% | 0.00% | 0.08s | 0 |
| `BM_04_red_stamp_200` | **Oracle Threshold** | 1.06 | 3.55% | 0.39% | 8.87s | 0 |
| `BM_04_red_stamp_200` | **wmclean (active)** | 1.32 | 50.66% | 1.44% | 9.55s | 211 |
| `BM_05_tiled_diag_150` | **No-Op (Identity)** | 21.17 | 0.44% | 0.00% | 0.02s | 0 |
| `BM_05_tiled_diag_150` | **Oracle Threshold** | 0.28 | 0.62% | 0.56% | 4.56s | 0 |
| `BM_05_tiled_diag_150` | **wmclean (active)** | 21.61 | 70.14% | 2.28% | 3.99s | 197 |
| `BM_06_camscanner_banner_200` | **No-Op (Identity)** | 31.62 | 0.09% | 0.00% | 0.03s | 0 |
| `BM_06_camscanner_banner_200` | **Oracle Threshold** | 1.09 | 3.57% | 0.29% | 8.85s | 0 |
| `BM_06_camscanner_banner_200` | **wmclean (active)** | 11.47 | 55.35% | 1.40% | 26.42s | 215 |
