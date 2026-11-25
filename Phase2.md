# Phase 2: Machine Learning Classification, Cross-PVT Analysis & Research Publication Guide

**Project:** BTP — Automated 6T SRAM Fault Injection, PVT Simulation & Diagnostic Framework  
**Document Purpose:** Complete, step-by-step handover and execution guide for collaborators taking over Phase 2.  
**Repository State:** The 45-corner dataset, manuscript, and reproduction script are available.

---


## 1. Completed simulation work

- The [healthy core](circuits/core/cell_core_healthy.net) uses 180 nm Level-1 MOS models. The geometric cell ratio is 1.333 and the pull-up ratio is 0.667.
- [SNM extraction](src/snm_extraction.py) searches for the largest axis-aligned square in each butterfly lobe on a finite grid with interpolation. The [nominal manual check](validation/snm_manual_check/results.md) reports 470.06 mV for hold and 263.69 mV for read.
- [Fault injection](src/fault_injection.py) inserts a series resistance on either access path, a bridge between $Q$ and $Q_b$, or threshold drift on the storage or access devices. The six drift levels increase $|V_{\mathrm{TO}}|$ by 5%, 10%, 15%, 20%, 25%, or 30%. The storage PMOS threshold becomes more negative.
- The benchmark uses TT, FF, and SS, $V_{DD}=0.9$, $1.0$, or $1.1$ V, and temperatures of $-40$, $0$, $27$, $75$, or $125^\circ$C. Its 45 corners contain 30 observations each. Seven operations per target give 9,450 SPICE runs, plus 315 for the matched healthy references.
- [The saved dataset](data/sram_fault_dataset.csv) contains 1,350 rows and 34 columns. The nominal TT, 1.0 V, 27 °C export contains 30 rows. The remaining-corner export contains 1,320 rows. The full dataset includes numerical fallbacks, and 23 rows are marked nonconverged.

## 2. Manuscript evaluation

The [paper](Research_paper.tex) compares Random Forest, HistGradientBoosting, Decision Tree, and Logistic Regression. In each of the 45 LOCO folds, the models train on 44 corners (1,320 rows) and test on the remaining corner (30 rows). The same four functional flags are included in each 12-input feature set. The raw set adds eight electrical measurements. The normalized set adds five ratios to a healthy reference and three internal ratios.

The separate development pipeline in [src/ml_pipeline.py](src/ml_pipeline.py) also supports a nominal 80/20 split, Linear SVM, and a rule-based detector. Its saved models and aggregate summaries use the development feature sets and model settings. Historical 15-corner audit results are under [reports/audit](reports/audit/).

## 3. Reproduce the reported checks

Run from the project root after installing the [project dependencies](README.md#1-prerequisites--environment-setup). Install `shap` for the second command.

```bash
python scripts/reproduce_manuscript_checks.py
python scripts/reproduce_manuscript_checks.py --rf --shap
python scripts/verify_checkpoints.py
```

The first command reproduces Table II from the CSV. The second also fits the Random Forest controls and recomputes Tree SHAP. The third regenerates the [validation report](reports/validation_checkpoints_report.md) from the saved dataset, the Table III CSV, and nominal deck assembly checks. Use the [README](README.md#quickstart) for environment setup and simulator configuration.

### Manuscript figures

| Figure | What it shows |
|---|---|
| [Fig. 3](fig1_feature_dispersion_comparison.png) | Raw and normalized CVs for the 5 kΩ open on each access device. |
| [Fig. 4](fig2_cross_pvt_loco_accuracy.png) | Pooled and worst-corner accuracy for the paper's four models. Random Forest gains 24.22 percentage points in pooled accuracy. |
| [Fig. 5](fig3_rf_pvt_corner_heatmap.png) | Random Forest accuracy at the 45 held-out corners. The normalized set has a 90.00% worst-corner floor. |
| [Fig. 8](fig4_shap_summary.png), [Fig. 9](fig4b_shap_class_importance.png) | Global and class-specific mean absolute Tree SHAP scores for the full-data Random Forest. |

Root figures are the manuscript assets. Development plotting scripts may write their own figures or replace files with matching names. Use the reproduction script above to check metrics without regenerating publication figures.

---

## 4. Key Results & Expected Numbers

Use the reproduction script to check these benchmark results.

### 4.1 Feature Dispersion (Healthy Cells across 45 Corners)
*(From Table II of the research paper; CV is $100s/\bar{x}$ using $n-1$ sample standard deviation)*

| Feature | Raw CV (%) | Normalized CV (%) | Dispersion Reduction | Healthy Converged (43 Corners)<br>Raw $\to$ Norm CV |
|---|:---:|:---:|:---:|:---:|
| **Hold SNM ($\text{SNM}_{\text{hold}}$)** | $12.40\%$ | **$0.00\%$** | **$100\%$ eliminated** | $12.31\% \to \mathbf{0.00\%}$ |
| **Bitline Swing ($\Delta V_{\text{BL}}$)** | $38.50\%$ | **$0.00\%$** | **$100\%$ eliminated** | $38.83\% \to \mathbf{0.00\%}$ |
| **Write Delay ($t_{\text{write}}$)** | $18.30\%$ | **$3.49\%$** | **$80.9\%$ reduction** | $18.53\% \to \mathbf{0.00\%}$ |
| **Standby Current ($I_{\text{DDQ}}$)** | $197.59\%$ | **$111.80\%$** | **$43.4\%$ reduction** | $139.95\% \to \mathbf{0.00\%}$ |
| **Read SNM ($\text{SNM}_{\text{read}}$)** | $22.10\%$ | **$15.25\%$** | **$31.0\%$ reduction** | $22.26\% \to \mathbf{15.61\%}$ |

### 4.2 Classification Performance (Random Forest)
*(From Table III of the research paper; standard deviations reflect population std with $\mathrm{ddof}=0$)*

| Evaluation Protocol | Raw Features Accuracy | Normalized Feature Set | Generalization Gain | Worst-Corner Floor |
|---|:---:|:---:|:---:|:---:|
| **Leave-One-Corner-Out (LOCO across 45 corners)** | $75.11\% \pm 14.15\%$ [Min: 50.0%] | **$99.33\% \pm 1.94\%$** [Min: 90.0%] | **$+24.22\,\text{pp}$** | $50.00\% \to \mathbf{90.00\%}$ *(39/45 perfect)* |

On 1,327 rows marked converged, Random Forest scores 75.28% with raw features and 99.70% with normalized features. This subset retains the flagged read-SNM zero at TT, 0.9 V, 0 °C. Omitting all 30 rows at that corner before forming folds gives 99.32% on the remaining 44 corners.

---


## 5. Manuscript and supporting results

- [Research paper](Research_paper.tex) and [compiled PDF](SRAM_Research_paper.pdf) contain the circuit, dataset, evaluation, and limitations.
- [Table III benchmark CSV](reports/final_results/table3_loco_benchmark.csv) records the four-model LOCO comparison. Table IV covers the Random Forest controls.
- Table V scores only faulty observations at held-out parameter ranks. The $S_1$ fold scores 200/225, or 88.89%. The joint corner and parameter holdout scores 446/450, or 99.11%.
- Table VI reports class-specific and global Tree SHAP means from the fitted Random Forest.

## 6. Handover checklist

- Reproduce Table II and the manuscript's Random Forest sensitivity checks from the saved dataset.
- Regenerate the checkpoint report when the dataset or the stored benchmark summary changes.
- Use the manuscript source, the compiled PDF, and the corresponding Overleaf archive for submission.

---

## 7. Contact & Assistance
If any unexpected test failures or SPICE anomalies occur:
- Check `convergence_flags_*.csv` and `ref_sim_flags_*.csv` in [logs](logs/) for warnings from new runs. The [unsuffixed convergence log](logs/convergence_flags.csv) covers an earlier run.
- Check that your local `config.yaml` points to a valid LTspice executable path.
- Review [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) for deep circuit and algorithm specifications.
