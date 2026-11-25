# Automated 6T SRAM Fault Injection, PVT Simulation & Diagnostic Framework

[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](#2-run-test-suite)
[![Technology](https://img.shields.io/badge/Technology-180nm%20CMOS-blue.svg)](PROJECT_DOCUMENTATION.md#3-circuit-implementation--sizing-definitions)
[![Simulator](https://img.shields.io/badge/Simulator-LTspice-orange.svg)](#ltspice-configuration)
[![PVT Corners](https://img.shields.io/badge/PVT%20Corners-45%20Grid-purple.svg)](PROJECT_DOCUMENTATION.md#51-pvt-sweep-grid-45-corners)

This project injects parametric defects into a 6T SRAM cell, simulates them across 45 process-voltage-temperature (PVT) corners, and computes reference-normalized features for leave-one-corner-out (LOCO) diagnosis. The stored benchmark has 1,350 configuration–corner observations. The [manuscript](Research_paper.tex) describes the benchmark and its limitations.

---

## Key Features

1. **Decoupled Netlist Architecture**: Strict separation of the 6T core template (`circuits/core/`) and operational stimulus fragments (`circuits/testbenches/`).
2. **5-Class Fault Taxonomy**:
   - **Class 0 (Healthy):** Nominal baseline cell.
   - **Class 1 (Resistive Open).** A series resistance on one access path, $M_5$ or $M_6$ ($200\,\Omega$ to $10\,\text{k}\Omega$).
   - **Class 2 (Bridging Fault):** Shunt resistance between internal storage nodes $Q$ and $\bar{Q}$ ($500\,\Omega$ to $10\,\text{k}\Omega$).
   - **Class 3 ($V_{th}$ Drift Storage).** Increase $|V_{\mathrm{TO}}|$ by 5%, 10%, 15%, 20%, 25%, or 30% on $M_1$–$M_4$. PMOS $V_{\mathrm{TO}}$ becomes more negative.
   - **Class 4 ($V_{th}$ Drift Access).** Apply the same six magnitude increases to access NMOS $M_5$ and $M_6$.
3. **Multi-Operation Simulation**:
   - **Dynamic Transient:** `hold`, `write` (Write 1), `write_0` (Write 0), `read` (Read 1), `read_0` (Read 0).
   - **Static DC Butterfly Curve.** `snm_hold` and `snm_read` use a numerical inscribed-square search with piecewise-linear interpolation.
4. **45-Corner PVT Operating Envelope**:
   - **Process Corners (3):** Typical-Typical (TT), Fast-Fast (FF), Slow-Slow (SS).
   - **Supply Voltages (3):** $0.9\,\text{V}$ ($-10\%$), $1.0\,\text{V}$ (nominal), $1.1\,\text{V}$ ($+10\%$).
   - **Temperatures (5):** $-40^\circ\text{C}$ (industrial cold), $0^\circ\text{C}$, $27^\circ\text{C}$ (room nominal), $75^\circ\text{C}$, $125^\circ\text{C}$ (worst-case thermal).
   - Total PVT corners = $3 \times 3 \times 5 = 45$ corners.
5. **Reference normalization.** Compares five target measurements with a simulated healthy cell at the same corner and adds three ratios within the target measurements. Table II reports how much dispersion remains at a fixed $5\,\text{k}\Omega$ open.
6. **Cross-PVT evaluation.**
   - The manuscript compares Random Forest, HistGradientBoosting, Decision Tree, and Logistic Regression. Each of the 45 LOCO folds trains on 44 corners and tests one corner.
   - Random Forest accuracy rises from **75.11%** on raw features to **99.33%** on the normalized feature set. Its worst-corner accuracy rises from 50.00% to 90.00%. HistGradientBoosting reaches 99.41% with the normalized set.
   - The development pipeline and historical 15-corner audit retain separate baseline and Linear SVM experiments. See the [project documentation](PROJECT_DOCUMENTATION.md#8-machine-learning-results--cross-pvt-validation).

---

## Quickstart

### 1. Prerequisites & Environment Setup

Install dependencies:
```bash
pip install PyLTSpice pyyaml pandas numpy scipy scikit-learn matplotlib seaborn pytest
```

### 2. Run Test Suite

```bash
pytest -v
```
The suite includes LTspice integration tests. Configure the executable path before running those tests.

### 3. Reproduce manuscript checks

Use the checked-in dataset to reproduce the Table II statistics and regenerate the validation report.
```bash
python scripts/reproduce_manuscript_checks.py
python scripts/verify_checkpoints.py
```

Use `python scripts/reproduce_manuscript_checks.py --rf --shap` to refit the Random Forest controls and recompute the SHAP rankings. Install `shap` to run that option. The [saved Table III summary](reports/final_results/table3_loco_benchmark.csv) contains the four-model benchmark results.

### 4. Generate the 45-Corner Dataset

```bash
python src/generate_dataset.py
```
The full sweep executes 9,765 SPICE simulations across worker threads, with 9,450 target operation decks and 315 healthy reference decks. It exports:
- `data/sram_fault_dataset.csv` (1,350 rows $\times$ 34 columns, 0 NaNs)
- `data/sram_fault_dataset_indist.csv` (30 rows, nominal corner: TT, 1.0V, 27°C)
- `data/sram_fault_dataset_ood.csv` (1,320 rows across the 44 non-nominal corners)
- `logs/convergence_flags_*.csv` (traceability logs)
- `logs/ref_sim_flags_*.csv` (reference warning logs)

---

## Repository Structure

```
btp-sram-fault-diagnosis/
├── circuits/
│   ├── core/              # cell_core_healthy.net (M1-M6 180nm core)
│   ├── reference/         # Golden reference Hold & Read SNM decks
│   └── testbenches/       # Stimulus fragments (hold, write, read, snm)
├── src/
│   ├── fault_injection.py      # Open, Bridge, and Vth Drift injection
│   ├── testbench_composer.py   # Netlist composer & PVT scaling engine
│   ├── simulation_runner.py    # Multi-threaded headless LTspice runner
│   ├── snm_extraction.py       # Pure NumPy largest inscribed square search
│   ├── feature_extractor.py    # Dynamic timing, sense voltage & SNM extraction
│   ├── generate_dataset.py     # 45-corner dataset generation orchestrator
│   ├── baseline_detector.py    # Non-ML 3-sigma statistical baseline detector
│   ├── ml_pipeline.py          # Classical ML classifiers & LOCO evaluation
│   └── cross_pvt_analysis.py   # Development dispersion analysis and plots
├── data/
│   ├── sram_fault_dataset.csv         # Master dataset (1350 x 34, zero NaNs)
│   ├── sram_fault_dataset_indist.csv  # Nominal in-distribution split
│   └── sram_fault_dataset_ood.csv     # Non-nominal export (44 corners, 1320 samples)
├── scripts/
│   ├── reproduce_manuscript_checks.py # Recompute dispersion, RF checks, and SHAP
│   ├── verify_checkpoints.py          # Regenerate the validation report
│   ├── verify_dataset_physics.py      # Descriptive checks of stored measurements
│   ├── demo_feature_extraction.py     # 25-sim dynamic transient benchmark
│   └── demo_snm_features.py           # Butterfly-curve demonstration
├── tests/                             # Unit and LTspice integration checks
├── config.example.yaml                # Committed template for local config
└── PROJECT_DOCUMENTATION.md           # Exhaustive technical documentation
```

---

## LTspice Configuration

LTspice is required for running new SPICE simulations. Path resolution is automatic on first execution:
`src/simulation_runner.py` automatically scans standard paths on Windows across all drive letters (`C:\Program Files\ADI\LTspice\LTspice.exe`, `C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe`, etc.) and writes the discovered executable path to `config.yaml`.

If you have a custom install location, create `config.yaml` from `config.example.yaml`:
```yaml
ltspice_path: 'C:\Program Files\ADI\LTspice\LTspice.exe'
```

---

## License

This project was developed for the B.Tech Project (BTP) on SRAM Fault Diagnosis.
