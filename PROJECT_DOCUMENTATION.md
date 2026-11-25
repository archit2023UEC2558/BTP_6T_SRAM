# 6T SRAM Fault Injection, Simulation & Diagnostic Pipeline — Complete Project Documentation

- **Project.** BTP — Automated SRAM Fault Injection & Diagnosis.
- **Circuit.** 180 nm 6T SRAM with generic Level-1 models in LTspice.
- **Dataset.** 1,350 configuration–corner observations across 45 PVT corners. The 34-column CSV stores finite values, including numerical fallbacks.
- **Models.** The paper benchmarks Random Forest, HistGradientBoosting, Decision Tree, and Logistic Regression.
- **Random Forest LOCO.** Pooled accuracy rises from 75.11% to 99.33%, a gain of 24.22 percentage points. Worst-corner accuracy rises from 50.00% to 90.00%.

---

## 1. Executive Summary & Architecture

This repository contains an end-to-end automated framework for:
1. Modeling physical and parametric defects in a standard 6T SRAM cell (Resistive Opens, Bridging Faults, and $V_{th}$ Drifts).
2. Composing runnable SPICE testbenches for multiple memory operations (Hold, Write 1/0, Read 1/0, and DC Butterfly-Curve Hold/Read Static Noise Margins).
3. Simulating across a 45-corner process-voltage-temperature (PVT) grid with three process settings (TT, FF, SS), three supply voltages (0.9, 1.0, and 1.1 V), and five temperatures (−40, 0, 27, 75, and 125 °C).
4. Extracting raw dynamic/static features and computing reference-normalized features using a separately simulated healthy cell at the same corner.
5. Comparing the four manuscript classifiers and keeping the development pipeline available for the historical Linear SVM and rule-based experiments.
6. Evaluating cross-PVT generalization with leave-one-corner-out (LOCO) folds over all 45 corners and separate fault-parameter holdouts.
7. Conducting data policy sensitivity audits and generating publication-ready figures.

```
       +----------------------------+
       | cell_core_healthy.net      |
       +----------------------------+
                     |
         [ src/fault_injection.py ]
                     |
       +----------------------------+       +------------------------------------+
       |   Faulty Cell Core         |  +    |  Stimulus Fragment                 |
       | (Open / Bridge / Vth Drift)|       | (HOLD / WRITE 1/0 / READ 1/0 / SNM)|
       +----------------------------+       +------------------------------------+
                     \                                 /
                      \                               /
                       [ src/testbench_composer.py ]
                             (PVT Parameterization)
                             (duplicate if SNM)
                                      |
                     +----------------------------------+
                     | Composed Runnable Testbench (.net)|
                     +----------------------------------+
                                      |
                     [ src/simulation_runner.py ]
                        (Multi-Threaded LTspice)
                                      |
                     +----------------------------------+
                     | Raw Waveforms (.raw) & Logs (.log)|
                     +----------------------------------+
                                      |
                     [ src/feature_extractor.py ]
                     [ src/snm_extraction.py    ]
                                      |
                     +----------------------------------+
                     | Master Dataset (1350 x 34 CSV)   |
                     |  Raw + Physics-Normalized Ratios |
                     +----------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
           [ src/baseline_detector.py ]       [ src/ml_pipeline.py ]
           (3-Sigma Healthy Rule Checks)      (Development experiments)
                     |                                 |
                     +----------------+----------------+
                                      |
                         [ src/cross_pvt_analysis.py ]
                         [ scripts/redraw_*.py       ]
                         (Development reports and figures)
```

---

## 2. Directory Structure

```
btp-sram-fault-diagnosis/
├── circuits/
│   ├── core/
│   │   └── cell_core_healthy.net        # Validated M1-M6 core template (symmetric access devices)
│   ├── reference/
│   │   ├── snm_read_healthy.net         # Golden read SNM reference deck
│   │   └── snm_hold_healthy.net         # Golden hold SNM reference deck
│   └── testbenches/
│       ├── stimulus_hold.net            # HOLD stimulus (retention & IDDQ test)
│       ├── stimulus_write.net           # WRITE 1 stimulus (BL=1V, BLB=0V; exercises M6)
│       ├── stimulus_write_0.net         # WRITE 0 stimulus (BL=0V, BLB=1V; exercises M5)
│       ├── stimulus_read.net            # READ 1 stimulus (precharged BL/BLB; exercises M6)
│       ├── stimulus_read_0.net          # READ 0 stimulus (precharged BL/BLB; exercises M5)
│       ├── stimulus_snm_hold.net        # Hold SNM stimulus (DC sweep V5, WL=0V, ideal BL/BLB)
│       └── stimulus_snm_read.net        # Read SNM stimulus (DC sweep V5, WL=1V, ideal BL/BLB)
├── src/
│   ├── utils.py                         # File I/O helpers (load_cell_core, save_cell_core)
│   ├── fault_injection.py               # Fault injection algorithms (Open, Bridge, Vth Drift)
│   ├── testbench_composer.py            # Netlist assembly, PVT parameterization, core duplication
│   ├── simulation_runner.py             # Multi-threaded direct batch runner, DC integrity validation
│   ├── snm_extraction.py                # Pure NumPy largest inscribed square search (Hold & Read)
│   ├── feature_extractor.py             # Dynamic timing, differential voltage, IDDQ & SNM extraction
│   ├── generate_dataset.py              # Full 45-corner dataset generation orchestrator
│   ├── baseline_detector.py             # Non-ML 3-sigma statistical baseline classifier
│   ├── ml_pipeline.py                   # Classical ML models, Standard Split, and LOCO protocols
│   └── cross_pvt_analysis.py            # Feature spread analysis and publication-ready figures
├── models/                              # 20 development-pipeline pickles from two feature sets and two protocols
├── data/
│   ├── sram_fault_dataset.csv           # Master complete dataset (1350 rows x 34 cols, 0 NaNs)
│   ├── sram_fault_dataset_indist.csv    # In-distribution nominal dataset (30 rows, TT, 1.0V, 27°C)
│   ├── sram_fault_dataset_ood.csv       # Non-nominal export (1320 rows across 44 corners)
│   └── demo_features/                   # Transient and SNM benchmark demonstration outputs
├── logs/
│   ├── convergence_flags.csv            # Earlier target warning log
│   └── ref_sim_flags_FF.csv              # Example process-specific reference warning log
├── reports/
│   ├── final_results/
│   │   ├── comparison_table.csv         # Development-pipeline evaluation across protocols
│   │   ├── loco_summary.csv             # Development-pipeline LOCO summary
│   │   ├── table3_loco_benchmark.csv    # Four-model manuscript summary
│   │   ├── feature_spread_comparison.csv# Development dispersion output
│   │   └── fault_feature_spread.csv     # Per-fault-class CV dispersion metrics
│   ├── figures/
│   │   ├── fig1_feature_dispersion_comparison.png  # Development copy of the dispersion plot
│   │   ├── fig1b_fault_dispersion.png              # Cross-corner CV bar plots across fault classes
│   │   ├── fig2_cross_pvt_loco_accuracy.png        # Bar chart comparing Raw vs Invariant (LOCO)
│   │   ├── fig3_rf_pvt_corner_heatmap.png          # Per-corner Random Forest accuracy heatmap
│   │   └── fig3b_rf_grid.png                       # 3x5 voltage-temperature grid heatmap
│   └── audit/                           # Data policy sensitivity analysis
│       ├── dataset_clean_main.csv       # 428-row dataset (excluding 22 waveform timeouts)
│       ├── dataset_clean_strict.csv     # 302-row dataset (strictly converged==1)
│       ├── excluded_rows.csv            # Breakdown of excluded simulation rows
│       ├── nonconverged_by_reason.csv   # Failure taxonomy breakdown
│       ├── loco_summary_original_450.csv # Historical 15-corner summary
│       ├── final_comparison_table.csv   # Historical LOCO comparison across row policies
│       └── final_comparison_figure.png  # Accuracy progression across dataset policies
├── scripts/
│   ├── reproduce_manuscript_checks.py   # Recompute Table II, RF controls, and Tree SHAP
│   ├── verify_checkpoints.py            # Regenerate the 45-corner validation report
│   ├── verify_dataset_physics.py        # Exploratory feature summaries
│   ├── apply_row_policy.py              # Filters dataset by simulation validity policies
│   ├── audit_nonconverged.py            # Deep-dive log parser for non-monotonic/timeout decks
│   ├── make_final_comparison.py         # Historical 15-corner comparative aggregator
│   ├── redraw_fig1_faults.py            # Re-draws Fig 1b fault-level dispersion
│   ├── redraw_fig3_grid.py              # Re-draws Fig 3b voltage-temperature grid
│   ├── demo_feature_extraction.py       # 25-sim symmetric dynamic benchmark
│   └── demo_snm_features.py             # Butterfly-curve Hold & Read SNM demonstration
├── tests/                               # Python and LTspice integration tests
├── validation/
│   └── snm_manual_check/
│       └── results.md                   # Golden validation measurements & GUI cross-checks
├── Phase2.md                            # Comprehensive Phase 2 handover specification
├── config.example.yaml                  # Template for local LTspice path configuration
├── pytest.ini                           # Test suite configuration
└── README.md                            # Quickstart & user documentation
```

---

## 3. Circuit Implementation & Sizing Definitions

### 3.1 6T SRAM Cell Topology ([`circuits/core/cell_core_healthy.net`](circuits/core/cell_core_healthy.net))

The storage cell consists of two cross-coupled CMOS inverters ($M_1/M_4$ at $Q_b$ and $M_2/M_3$ at $Q$) and two NMOS access pass-gates ($M_5$ and $M_6$). Symmetrical orientation is enforced: drain = internal storage node, source = bitline.

```spice
M1 Qb Q Vdd Vdd SRAM_PMOS l=180nm w=360nm
M2 Q Qb Vdd Vdd SRAM_PMOS l=180nm w=360nm
M3 Q Qb 0 0 SRAM_NMOS l=180nm w=720nm
M4 Qb Q 0 0 SRAM_NMOS l=180nm w=720nm
M5 Q WL BL 0 SRAM_NMOS l=180nm w=540nm
M6 Qb WL BLB 0 SRAM_NMOS l=180nm w=540nm
Cq  Qb 0 1fF
Cqb Q  0 1fF
.model SRAM_NMOS NMOS (LEVEL=1 VTO=0.45 KP=250u GAMMA=0.4 LAMBDA=0.08 PHI=0.7 TOX=4n)
.model SRAM_PMOS PMOS (LEVEL=1 VTO=-0.45 KP=100u GAMMA=0.4 LAMBDA=0.1 PHI=0.7 TOX=4n)
```

- **Transistor Sizing Ratios:**
  - Pull-down NMOS ($M_3, M_4$): $W = 720\,\text{nm}$
  - Access NMOS ($M_5, M_6$): $W = 540\,\text{nm}$ ($\text{Cell Ratio } \beta = 720/540 = 1.33$)
  - Pull-up PMOS ($M_1, M_2$): $W = 360\,\text{nm}$ ($\text{Pull-Up Ratio } \gamma = 360/540 = 0.667$)
  - Gate Length ($L$): $180\,\text{nm}$ across all devices

### 3.2 Dynamic & Static Stimulus Fragments
- **HOLD (`stimulus_hold.net`):** $WL = 0\,\text{V}, BL = BLB = V_{DD}$. Retention check and standby current ($I_{\text{ddq}}$) measurement.
- **WRITE 1 (`stimulus_write.net`):** $BL = V_{DD}, BLB = 0\,\text{V}$. $M_6$ pulls $Q_b$ down against pull-up $M_1$.
- **WRITE 0 (`stimulus_write_0.net`):** $BL = 0\,\text{V}, BLB = V_{DD}$. $M_5$ pulls $Q$ down against pull-up $M_2$.
- **READ 1 (`stimulus_read.net`):** Precharged bitlines ($C_{\text{BL}} = 20\,\text{fF}$ charged to $V_{DD}$). $BLB$ discharges through $M_6/M_4$.
- **READ 0 (`stimulus_read_0.net`):** Precharged bitlines ($C_{\text{BL}} = 20\,\text{fF}$ charged to $V_{DD}$). $BL$ discharges through $M_5/M_3$.
- **SNM HOLD (`stimulus_snm_hold.net`):** Access devices OFF ($WL = 0\,\text{V}$). DC sweep $V_5$ from $0 \to V_{DD}$ in $1\,\text{mV}$ steps. Traces $V(Q), V(Qb), V(Q\_c2), V(Qb\_c2)$.
- **SNM READ (`stimulus_snm_read.net`):** Access devices ON ($WL = V_{DD}$, $BL = BLB = V_{DD}$). Same $1\,\text{mV}$ DC sweep under wordline activation.

---

## 4. Fault Taxonomy & Physics

| Class | Fault Type | Physical Mechanism | Target Devices | Severity Sweep |
|---|---|---|---|---|
| **0** | **Healthy** | Nominal defect-free cell | None | One healthy observation per corner |
| **1** | **Resistive Open** | Contact/via void or metal electromigration | Access pass-gates $M_5$ or $M_6$ | $200\,\Omega, 500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **2** | **Bridging Fault** | Inter-layer dielectric breakdown / metal sliver | Storage nodes $Q \leftrightarrow \bar{Q}$ | $500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **3** | **$V_{th}$ Drift (Storage)** | Bias Temperature Instability (BTI) / HCI | Inverter pair $M_1$–$M_4$ only | $\lvert V_{\mathrm{TO}}\rvert$ increases by 5%, 10%, 15%, 20%, 25%, or 30% |
| **4** | **$V_{th}$ Drift (Access)** | Pass-gate dielectric trapping | Access pair $M_5/M_6$ only | $\lvert V_{\mathrm{TO}}\rvert$ increases by 5%, 10%, 15%, 20%, 25%, or 30% |

---

At an affected PMOS, increasing $|V_{\mathrm{TO}}|$ makes $V_{\mathrm{TO}}$ more negative.

## 5. PVT Simulation Engine & Multi-Threading


### 5.1 PVT Sweep Grid (45 Corners)

- **Process settings** — TT, FF, SS. FF applies $0.90\times |V_{\mathrm{TO}}|$ and $1.10\times KP$ relative to TT. SS applies $1.10\times |V_{\mathrm{TO}}|$ and $0.90\times KP$.
- **Supply voltage** — $V_{DD}=0.9, 1.0, 1.1\,\text{V}$.
- **Temperature** — $T=-40, 0, 27, 75, 125\,^\circ\text{C}$.
- **Grid** — $3\times3\times5=45$ corners and 30 observations per corner, for 1,350 rows.
- **Nominal export** — TT at 1.0 V and 27 °C (`TT_C_27C_1P0V`), 30 rows. The remaining-corner export has 1,320 rows across 44 corners.
- **SPICE runs** — 9,450 target-operation runs and 315 matched healthy reference-operation runs, for 9,765 total.

### 5.2 Dynamic Netlist Composer
[`src/testbench_composer.py`](src/testbench_composer.py) dynamically updates:
1. Voltage sources (`Vdd`, `V1`, `V3`, `V4`) to target $V_{DD}$.
2. Wordline and bitline pulse amplitude levels.
3. Precharge initial condition `.ic` voltages ($V(BL) = V_{DD}, V(Q) = V_{DD}$).
4. SNM DC sweep range (`.dc V5 0 <vdd> 1m`).
5. Injects `.temp <T>` directive.

### 5.3 Direct Process Runner & Race Condition Resolution
PyLTSpice's default `SimRunner.run_now()` exhibits file collision errors on Windows when multiple Python threads access shared deck files. To solve this, [`src/simulation_runner.py`](src/simulation_runner.py) directly invokes headless LTspice processes (`LTspice.exe -b -ascii <unique_deck>`). The dataset generator supports pruning intermediate waveform files.

---

## 6. Dataset Schema & Physics Normalization

### 6.1 Master Dataset Schema (34 Columns)
Each row in [`data/sram_fault_dataset.csv`](data/sram_fault_dataset.csv) contains:
- **Metadata:** `sample_id`, `corner_id`, `pvt_type`, `vdd_v`, `temperature_c`, `process_corner`
- **Labels:** `fault_class` ($0$ to $4$), `fault_name`, `fault_target`, `severity_value`
- **Raw Dynamic Features:** `i_ddq_uA`, `t_write_ps`, `i_write_peak_mA`, `dv_bl_strobe_mV`, `t_sense_ps`, `v_bump_mV`, `hold_pass`, `write_pass`, `read_stable_pass`
- **Raw Static & SNM Features:** `snm_hold_v`, `snm_read_v`, `snm_asym_hold_v`, `snm_asym_read_v`, `snm_drop_v`, `is_bistable`
- **Reference-normalized and internal ratios**
  - $\text{iddq\_norm} = I_{\text{ddq}} / I_{\text{ddq, ref}}$
  - $\text{t\_write\_norm} = t_{\text{write}} / t_{\text{write, ref}}$
  - $\text{dv\_bl\_norm} = \Delta V_{\text{BL}} / \Delta V_{\text{BL, ref}}$
  - $\text{snm\_hold\_norm} = \text{SNM}_{\text{hold}} / \text{SNM}_{\text{hold, ref}}$
  - $\text{snm\_read\_norm} = \text{SNM}_{\text{read}} / \text{SNM}_{\text{read, ref}}$
  - Dimensionless internal ratios: $\text{read\_write\_delay\_ratio}$, $\text{snm\_ratio}$, $\text{dv\_bl\_vdd\_ratio}$
- **Row status.** `converged` is the stored row flag. The dataset has 1,327 rows marked converged and 23 marked nonconverged. Rows marked converged can retain extraction fallbacks.

---


## 7. Electrical Measurements and Dispersion

Use the [checkpoint report](reports/validation_checkpoints_report.md) for the stored nominal values and [the SNM reference check](validation/snm_manual_check/results.md) for the nominal DC sweeps. [The exploratory feature script](scripts/verify_dataset_physics.py) groups the saved measurements by operating conditions and fault parameters.

### 7.1 Sampled standby current

At TT and 1.0 V, the stored healthy current is 6.2, 4.8, 4.2, 855.3, and 502.0 nA for $-40$, $0$, $27$, $75$, and $125^\circ$C, respectively. These samples come from the saved hold transient at 20 ns. The nonmonotonic trend and uncalibrated Level-1 models limit their interpretation as leakage measurements.

### 7.2 Nominal timing and read measurements

At TT, 1.0 V, and 27 °C, the stored healthy row gives a 224.8 mV mean bitline differential at the 2.15 ns strobe, 60.0 ps mean sense delay, 125.0 mV maximum read bump, and 60.2 ps mean write delay.

### 7.3 Fault signatures and numerical fallbacks

Series resistance changes the access path, a bridge couples the storage nodes, and threshold-magnitude shifts alter storage or access device drive. The benchmark measures timing, swing, current, and SNM responses to these injections. Interpret very long delays and zero SNM alongside the row flags and the fallback policy in Section IV-C of the manuscript.

### 7.4 Dispersion Reduction Across 45 PVT Corners
Table II compares sample CVs before and after normalization against the matched simulated references. The fixed-open columns use rows marked converged for each injection site.

| Feature | Healthy All (45 Corners)<br>Raw CV $\to$ Normalized CV | Healthy Converged (43 Corners)<br>Raw CV $\to$ Normalized CV | 5k$\Omega$ Open $M_5$ (43 Corners)<br>Raw CV $\to$ Normalized CV | 5k$\Omega$ Open $M_6$ (41 Corners)<br>Raw CV $\to$ Normalized CV |
|---|:---:|:---:|:---:|:---:|
| **Hold SNM ($\text{SNM}_{\text{hold}}$)** | $12.40\% \to \mathbf{0.00\%}$ | $12.31\% \to \mathbf{0.00\%}$ | $12.49\% \to \mathbf{0.00\%}$ | $12.62\% \to \mathbf{0.00\%}$ |
| **Bitline Swing ($\Delta V_{\text{BL}}$)** | $38.50\% \to \mathbf{0.00\%}$ | $38.83\% \to \mathbf{0.00\%}$ | $36.38\% \to \mathbf{2.28\%}$ | $37.01\% \to \mathbf{2.30\%}$ |
| **Write Delay ($t_{\text{write}}$)** | $18.30\% \to \mathbf{3.49\%}$ | $18.53\% \to \mathbf{0.00\%}$ | $15.05\% \to \mathbf{3.66\%}$ | $17.28\% \to \mathbf{2.50\%}$ |
| **Standby Current ($I_{\text{DDQ}}$)** | $197.59\% \to \mathbf{111.80\%}$ | $139.95\% \to \mathbf{0.00\%}$ | $148.33\% \to \mathbf{0.00\%}$ | $142.69\% \to \mathbf{0.00\%}$ |
| **Read SNM ($\text{SNM}_{\text{read}}$)** | $22.10\% \to \mathbf{15.25\%}$ | $22.26\% \to \mathbf{15.61\%}$ | $22.54\% \to \mathbf{15.65\%}$ | $22.74\% \to \mathbf{16.04\%}$ |

---

## 8. Machine Learning Results & Cross-PVT Validation


### 8.1 Development pipeline and historical audit

[src/ml_pipeline.py](src/ml_pipeline.py) supports a nominal 80/20 split and LOCO on the supplied dataset. It runs Random Forest, HistGradientBoosting, Logistic Regression, Linear SVM, and a rule-based detector with its development feature sets. The 20 saved pickles correspond to five development models, two feature sets, and two protocols. The development comparison and LOCO summary files use those settings.

The [historical 15-corner audit](reports/audit/final_comparison_table.csv) compares the original 450-row dataset with the 428- and 302-row subsets. Its original summary is preserved in [loco_summary_original_450.csv](reports/audit/loco_summary_original_450.csv).

### 8.2 Manuscript benchmark across 45 corners
Iteratively trains on 44 corners ($1,320$ samples) and tests on 1 held-out corner ($30$ samples) across all 45 PVT corners (1,350 evaluations):

#### [Saved Table III benchmark results](reports/final_results/table3_loco_benchmark.csv)
*(Corresponds to Table III in the research paper; standard deviations reflect population standard deviation with $\mathrm{ddof}=0$)*

| Model Architecture | Raw Features Accuracy<br>(Mean $\pm$ Std) [Min, Max] | Normalized Feature Set<br>(Mean $\pm$ Std) [Min, Max] | Generalization<br>Gain (pp) | Worst-Corner Floor<br>(Raw $\to$ Normalized) |
|---|:---:|:---:|:---:|:---:|
| **Random Forest** | $75.11\% \pm 14.15\%$ [50.0%, 96.7%] | **$99.33\% \pm 1.94\%$** [**90.0%**, 100.0%] | **$+24.22\,\text{pp}$** | $50.00\% \to \mathbf{90.00\%}$ *(39/45 perfect corners)* |
| **HistGradientBoosting** | $77.56\% \pm 15.21\%$ [40.0%, 100.0%] | **$99.41\% \pm 2.14\%$** [**86.7%**, 100.0%] | **$+21.85\,\text{pp}$** | $40.00\% \to \mathbf{86.67\%}$ *(40/45 perfect corners)* |
| **Decision Tree** | $73.56\% \pm 12.42\%$ [36.7%, 96.7%] | **$98.44\% \pm 5.60\%$** [**63.3%**, 100.0%] | **$+24.89\,\text{pp}$** | $36.67\% \to \mathbf{63.33\%}$ |
| **Logistic Regression** | $80.59\% \pm 9.14\%$ [60.0%, 96.7%] | **$86.89\% \pm 7.08\%$** [**60.0%**, 96.7%] | **$+6.30\,\text{pp}$** | $60.00\% \to \mathbf{60.00\%}$ |


The manuscript compares two 12-input sets with the same four status flags. The raw set adds eight electrical measurements. The normalized set adds five ratios to the matched healthy reference and three internal ratios.

- **Table IV.** For Random Forest, the eight ratios alone give 99.48% LOCO accuracy and a 90.00% worst-corner accuracy. Electrical features can carry fallback or functional-failure information.
- **Table V.** The leave-one-severity-out (LOSO) experiment holds out parameter ranks and scores faulty observations. $S_1$ yields 200/225 (88.89%), and the weakest tested opens score 69/90 (76.67%). The joint corner and parameter holdout yields 446/450 (99.11%).

### 8.3 Sensitivity to simulation row flags

Filtering out the 23 rows marked nonconverged before splitting leaves 1,327 rows. Random Forest scores 75.28% with raw features and 99.70% with normalized inputs. The retained rows include the flagged healthy read-SNM zero at TT, 0.9 V, and 0 °C. Omitting the entire affected corner before splitting yields 99.32% for the normalized model across 44 corners.

Recompute these checks with [reproduce_manuscript_checks.py](scripts/reproduce_manuscript_checks.py) using `--rf`.

### 8.4 Explainability & Feature Attribution via TreeExplainer SHAP
Table VI reports mean absolute Tree SHAP values for the Random Forest fitted on all 1,350 rows with the normalized input set. The global value averages over rows and all five class outputs.

#### [Global Tree SHAP values](reports/final_results/table5_shap_importance.csv)
1. **`t_write_norm`** (0.0789). Ranks first globally.
2. **`dv_bl_norm`** (0.0776). Ranks second globally.
3. **`snm_read_norm`** (0.0567). Ranks third, including after the affected corner is omitted and the model refitted.
4. **`dv_bl_vdd_ratio`** (0.0269). The bitline-swing ratio to supply voltage.
5. **`snm_hold_norm`** (0.0263). The normalized hold margin.
6. **`iddq_norm`** (0.0249). The normalized sampled standby current.

---


## 9. Manuscript figures

The manuscript uses the root-level figures below. Artifacts in [reports/figures](reports/figures/) and [reports/audit](reports/audit/) also include development figures.

| Paper figure | Asset and content |
|---|---|
| 1 | [SRAM schematic](6T_SRAM_LTSPICE.jpg) |
| 2 | [Nominal hold butterfly curve and inscribed squares](snm_butterfly_diagram.png) |
| 3 | [Cross-corner CV for the fixed 5 kΩ opens](fig1_feature_dispersion_comparison.png) |
| 4 | [Pooled and worst-corner LOCO accuracy for four models](fig2_cross_pvt_loco_accuracy.png) |
| 5 | [Random Forest accuracy at 45 held-out corners](fig3_rf_pvt_corner_heatmap.png) |
| 6 | [Pooled LOCO confusion matrix](fig5_confusion_matrix.png) |
| 7 | [Row-wise feature perturbation and refitting](fig6_reference_mismatch_sensitivity.png) |
| 8, 9 | [Global](fig4_shap_summary.png) and [per-class](fig4b_shap_class_importance.png) Tree SHAP |

## 10. Automated checks

Run `python -m pytest` for the test suite. The LTspice integration cases need a configured simulator path. Run `python scripts/verify_checkpoints.py` to regenerate the [validation report](reports/validation_checkpoints_report.md) from the current dataset, nominal deck composer, and saved Table III results. The [manuscript reproduction script](scripts/reproduce_manuscript_checks.py) computes Table II and accepts `--rf` and `--shap` for the model checks.
