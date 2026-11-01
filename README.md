# Automated 6T SRAM Fault Injection, PVT Simulation & Diagnostic Framework

[![Tests](https://img.shields.io/badge/pytest-62%20passed-brightgreen.svg)](#running-tests)
[![Technology](https://img.shields.io/badge/Technology-180nm%20CMOS-blue.svg)](#circuit-implementation)
[![Simulator](https://img.shields.io/badge/Simulator-LTspice-orange.svg)](#ltspice-configuration)
[![PVT Corners](https://img.shields.io/badge/PVT%20Corners-45%20Grid-purple.svg)](#pvt-simulation-grid)

An automated, end-to-end pair-programming framework for injecting physical and parametric defects into a 6T SRAM cell, running multi-threaded SPICE simulations across a comprehensive 45-corner Process-Voltage-Temperature (PVT) grid (3 process corners $\times$ 15 V-T corners = 1,350 physical SPICE samples), computing physics-normalized invariant features, and training machine learning models with Leave-One-Corner-Out (LOCO) cross-PVT validation.

---

## Key Features

1. **Decoupled Netlist Architecture**: Strict separation of the 6T core template (`circuits/core/`) and operational stimulus fragments (`circuits/testbenches/`).
2. **5-Class Fault Taxonomy**:
   - **Class 0 (Healthy):** Nominal baseline cell.
   - **Class 1 (Resistive Open):** Symmetric contact/via opens on access transistors $M_5$ or $M_6$ ($200\,\Omega$ to $10\,\text{k}\Omega$).
   - **Class 2 (Bridging Fault):** Shunt resistance between internal storage nodes $Q$ and $\bar{Q}$ ($500\,\Omega$ to $10\,\text{k}\Omega$).
   - **Class 3 ($V_{th}$ Drift Storage):** Isolated threshold voltage shifts on cross-coupled inverters $M_1$–$M_4$ ($\pm 5\%$ to $\pm 30\%$).
   - **Class 4 ($V_{th}$ Drift Access):** Isolated threshold voltage shifts on access pass-gates $M_5/M_6$ ($\pm 5\%$ to $\pm 30\%$).
3. **Multi-Operation Simulation**:
   - **Dynamic Transient:** `hold`, `write` (Write 1), `write_0` (Write 0), `read` (Read 1), `read_0` (Read 0).
   - **Static DC Butterfly Curve:** `snm_hold` and `snm_read` with exact largest-inscribed-square search (pure NumPy).
4. **45-Corner PVT Operating Envelope**:
   - **Process Corners (3):** Typical-Typical (TT), Fast-Fast (FF), Slow-Slow (SS).
   - **Supply Voltages (3):** $0.9\,\text{V}$ ($-10\%$), $1.0\,\text{V}$ (nominal), $1.1\,\text{V}$ ($+10\%$).
   - **Temperatures (5):** $-40^\circ\text{C}$ (industrial cold), $0^\circ\text{C}$, $27^\circ\text{C}$ (room nominal), $75^\circ\text{C}$, $125^\circ\text{C}$ (worst-case thermal).
   - Total PVT corners = $3 \times 3 \times 5 = 45$ corners.
5. **Physics-Normalized Invariant Feature Engineering**: Normalizes transient delays, currents, and noise margins by a same-PVT healthy reference cell, reducing environmental parameter dispersion by up to $100\%$.
6. **Cross-PVT Machine Learning Pipeline**:
   - Classical classifiers: Random Forest, HistGradientBoosting, Decision Tree, Logistic Regression, and a Non-ML 3-sigma statistical baseline.
   - Evaluated under both **Standard Split** and **Leave-One-Corner-Out (LOCO)** cross-validation across all 45 corners.
   - Cross-PVT LOCO accuracy increases from **$75.11\%$** (raw features) to **$99.33\%$** (invariant features) for Random Forest, and to **$99.41\%$** for HistGradientBoosting, lifting the worst-case corner accuracy floor from $50.00\%$ to $90.00\%$.

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
All **62 unit tests** pass in under 10 seconds.

### 3. Verify Dataset Physical Sanity

Audit device physics trends (exponential thermal leakage, overdrive scaling, and dispersion reduction):
```bash
python scripts/verify_dataset_physics.py
```

### 4. Generate the 45-Corner Dataset

```bash
python src/generate_dataset.py
```
Executes 9,765 parallel SPICE simulations (9,450 fault runs + 315 dedicated reference characterizations) across worker threads and exports:
- `data/sram_fault_dataset.csv` (1,350 rows $\times$ 34 columns, 0 NaNs)
- `data/sram_fault_dataset_indist.csv` (30 rows, nominal corner: TT, 1.0V, 27°C)
- `data/sram_fault_dataset_ood.csv` (1,320 rows, 44 held-out PVT corners)
- `logs/convergence_flags_*.csv` (traceability logs)

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
│   └── cross_pvt_analysis.py   # Feature dispersion analysis & publication figures
├── data/
│   ├── sram_fault_dataset.csv         # Master dataset (1350 x 34, zero NaNs)
│   ├── sram_fault_dataset_indist.csv  # Nominal in-distribution split
│   └── sram_fault_dataset_ood.csv     # Held-out 44-corner OOD split (1320 samples)
├── scripts/
│   ├── verify_dataset_physics.py      # Automated physics validation audit
│   ├── demo_feature_extraction.py     # 25-sim dynamic transient benchmark
│   └── demo_snm_features.py           # Butterfly-curve demonstration
├── tests/                             # 62 unit tests covering all components
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
