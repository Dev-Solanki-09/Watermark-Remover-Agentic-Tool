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
| `BM_01_diag_gray_150` | **No-Op (Identity)** | 32.98 | 0.04% | 0.00% | 0.00s | 0 |
| `BM_01_diag_gray_150` | **Oracle Threshold** | 0.11 | 1.97% | 0.44% | 3.71s | 0 |
| `BM_01_diag_gray_150` | **wmclean (active)** | 32.98 | 9.54% | 0.07% | 86.33s | 452 |
| `BM_02_subtle_200` | **No-Op (Identity)** | 15.32 | 0.00% | 0.00% | 0.10s | 0 |
| `BM_02_subtle_200` | **Oracle Threshold** | 0.01 | 0.00% | 0.19% | 10.81s | 0 |
| `BM_02_subtle_200` | **wmclean (active)** | 15.32 | 0.00% | 0.00% | 4.93s | 454 |
| `BM_03_heavy_dark_200` | **No-Op (Identity)** | 82.05 | 0.09% | 0.00% | 0.15s | 0 |
| `BM_03_heavy_dark_200` | **Oracle Threshold** | 79.35 | 0.46% | 0.28% | 11.25s | 0 |
| `BM_03_heavy_dark_200` | **wmclean (active)** | 82.05 | 0.09% | 0.00% | 4.83s | 368 |
| `BM_04_red_stamp_200` | **No-Op (Identity)** | 54.74 | 0.01% | 0.00% | 0.35s | 0 |
| `BM_04_red_stamp_200` | **Oracle Threshold** | 1.06 | 3.55% | 0.39% | 7.44s | 0 |
| `BM_04_red_stamp_200` | **wmclean (active)** | 1.88 | 2.14% | 0.33% | 4.69s | 452 |
| `BM_05_tiled_diag_150` | **No-Op (Identity)** | 21.17 | 0.44% | 0.00% | 0.16s | 0 |
| `BM_05_tiled_diag_150` | **Oracle Threshold** | 0.28 | 0.62% | 0.56% | 3.88s | 0 |
| `BM_05_tiled_diag_150` | **wmclean (active)** | 21.17 | 9.91% | 0.07% | 2.90s | 357 |
| `BM_06_camscanner_banner_200` | **No-Op (Identity)** | 31.62 | 0.09% | 0.00% | 0.03s | 0 |
| `BM_06_camscanner_banner_200` | **Oracle Threshold** | 1.09 | 3.57% | 0.29% | 6.97s | 0 |
| `BM_06_camscanner_banner_200` | **wmclean (active)** | 31.62 | 0.09% | 0.00% | 2.47s | 458 |
