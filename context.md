# Context Dossier: PVT-Invariant Parametric Fault Diagnosis in 6T SRAM Cells Using Machine Learning

---

## 1. Project Overview & Administrative Details

- **Official Project Title:**  
  *PVT-Invariant Parametric Fault Diagnosis in 6T SRAM Cells Using Machine Learning*
- **Degree & Branch:**  
  Bachelor of Technology (B.Tech) in Electronics & Communication Engineering (ECE)
- **Institution:**  
  Netaji Subhas University of Technology (NSUT), New Delhi – 110078
- **Department:**  
  Department of Electronics & Communication Engineering
- **Supervisor:**  
  Dr. Shweta Gautam, Associate Professor, Department of ECE, NSUT
- **Student Team:**
  1. **Archit Gupta** — Roll No: `2023UEC2558`
  2. **Ayush Raj** — Roll No: `2023UEC2575`
  3. **Debasish Dhungana** — Roll No: `2023UEC2571`
  4. **Vansh Singh** — Roll No: `2023UEC2550`
- **Academic Milestone:**  
  Mid-Semester Evaluation (BTP Phase 1 / Mid-Sem Review, September–October 2026)
- **Manuscript:** [Reference-Normalized Parametric Fault Diagnosis Across PVT Corners in 6T SRAM Cells](Research_paper.tex).
- **Benchmark history:** The initial mid-semester comparison used 15 voltage-temperature corners. The manuscript uses 45 process-voltage-temperature corners.

---


## 2. Core Engineering Problem & Motivation

Process, voltage, and temperature can shift the electrical signatures used to diagnose SRAM faults. The study compares target measurements against a separately simulated healthy cell at the same operating corner. The goal is to distinguish five classes across held-out corners, including defects that can pass the selected functional checks. The benchmark uses generic Level-1 models and heuristic global process settings.

---

## 3. Circuit Sizing & Operational Physics

### Symmetrical 6T SRAM Cell Topology
- **Storage Inverters.** $M_1/M_4$ drive $Q_b$ and $M_2/M_3$ drive $Q$. The pull-up devices are $M_1/M_2$ and the pull-down devices are $M_3/M_4$.
- **Access Transistors:** NMOS pass-gates ($M_5, M_6$) gated by Wordline ($WL$), coupling internal nodes $Q$ and $Q_b$ to Bitlines ($BL, \overline{BL}$).
- **Aspect Ratios & Dimensions:**
  - Pull-up PMOS ($M_1, M_2$): $W = 360\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 2.0$
  - Driver NMOS ($M_3, M_4$): $W = 720\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 4.0$
  - Access NMOS ($M_5, M_6$): $W = 540\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 3.0$
- **Stability Ratio Criteria:**
  - **Cell Ratio (CR, $\beta$):** $\beta = \frac{(W/L)_{\text{driver}}}{(W/L)_{\text{access}}} = \frac{4.0}{3.0} = 1.333$ (geometric sizing heuristic).
  - **Pull-up Ratio (PR, $\gamma$):** $\gamma = \frac{(W/L)_{\text{pullup}}}{(W/L)_{\text{access}}} = \frac{2.0}{3.0} = 0.667$ (geometric sizing heuristic).

### Operational Modes & Verified Baselines (at Nominal $27^\circ\text{C}, 1.0\,\text{V}$)
1. **Hold.** With $WL=0$, the nominal stored observation has a sampled standby current of 4.2 nA at 20 ns. The nominal hold SNM is 470.06 mV.
2. **Read.** The nominal mean bitline differential is 224.8 mV at the fixed 2.15 ns strobe. The mean sense delay is 60.0 ps, the maximum read bump is 125.0 mV, and the read SNM is 263.69 mV. During Read 1, the complementary bitline discharges through $M_6$ and $M_4$.
3. **Write.** For Write 1, $BL=V_{DD}$ and $\overline{BL}=0$. With $WL$ asserted, $M_6$ pulls $Q_b$ down against $M_1$. The nominal mean delay across both write directions is 60.2 ps.

---

## 4. Parametric Fault Injection Taxonomy

Faults are programmatically injected into SPICE netlists via pure text replacement in `src/fault_injection.py`:

| Class | Fault Name | Physical Root Cause | SPICE Netlist Injection Mechanism | Injected Severity Sweep |
|:---:|---|---|---|---|
| **0** | **Healthy (Baseline)** | Nominal defect-free cell | None (Baseline reference netlist) | Nominal |
| **1** | **Resistive Open** | Contact/via void in access path | Series resistor `Rfault` inserted at $M_5$ or $M_6$ source/drain node | $200\,\Omega, 500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **2** | **$Q \leftrightarrow \overline{Q}$ Bridging** | Particle bridging between storage nodes | Parallel resistor `Rbridge` inserted across nodes $Q$ and $Q_b$ | $500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **3** | **$V_{\text{th}}$ Drift (Storage)** | NBTI/HCI aging on cross-coupled latch | Override `.model` parameter `VTO` on $M_1$–$M_4$ | Increase $\lvert V_{\mathrm{TO}}\rvert$ by 5%, 10%, 15%, 20%, 25%, or 30% |
| **4** | **$V_{\text{th}}$ Drift (Access)** | HCI degradation on pass-gates | Override `.model` parameter `VTO` on $M_5/M_6$ | Increase $\lvert V_{\mathrm{TO}}\rvert$ by 5%, 10%, 15%, 20%, 25%, or 30% |

For the storage drift, increasing $|V_{\mathrm{TO}}|$ makes the PMOS threshold more negative. The [validation report](reports/validation_checkpoints_report.md) records nominal testbench assembly checks.

---

## 5. Simulation Matrix & Dataset Infrastructure

- **Headless SPICE Automation:** Automated subprocess execution of LTspice with automatic parsing of `.log` files for convergence strings ("time step too small", "singular matrix", "gmin stepping failed", "no convergence").
- **45-Corner Operating Grid:**
  - Voltages: $0.9\,\text{V}, 1.0\,\text{V}, 1.1\,\text{V}$
  - Temperatures: $-40^\circ\text{C}, 0^\circ\text{C}, 27^\circ\text{C}, 75^\circ\text{C}, 125^\circ\text{C}$
  - Process: Typical-Typical (TT), Fast-Fast (FF), Slow-Slow (SS)
- **Dataset scale.** 1,350 observations across 45 corners. The benchmark comprises 9,450 target-operation runs and 315 reference-operation runs.
- **Dataset status.** The saved CSV contains finite values, including fallbacks. It marks 1,327 rows converged and 23 nonconverged. At TT, 0.9 V, and 0 °C, a flagged read-SNM zero remains in a row marked converged.

---

## 6. Multi-Domain Feature Engineering

### Dynamic Electrical Metrics
- $I_{\text{DDQ}}$: Sampled standby supply current ($\mu\text{A}$).
- $t_{\text{write}}$: Write switching flip delay from $50\%$ $WL$ rise to internal node crossing $0.5 V_{\text{DD}}$ ($\text{ps}$).
- $I_{\text{write, peak}}$: Peak dynamic current during write switching ($\text{mA}$).
- $\Delta V_{\text{BL}}$: Bitline discharge voltage differential at sense strobe ($\text{mV}$).
- $t_{\text{sense}}$: Delay to the first 100 mV bitline-differential crossing after the wordline's 50% rise ($\text{ps}$).
- $V_{\text{bump}}$: Voltage divider bump on storing '0' node during read ($\text{mV}$).

### Static Noise Margin (SNM) Extraction via Butterfly Curves
- Overlapping Voltage Transfer Characteristics (VTCs) of inverters: $V(Q_b) = f(V_Q)$ and $V(Q) = g(V_{Q_b})$.
- **Inscribed Square Algorithm:** Implemented in pure NumPy (`src/snm_extraction.py`). Computes the side length of the largest axis-aligned square inscribed in the smaller butterfly curve lobe:
  $$\text{SNM} = \min(\text{Lobe}_1, \text{Lobe}_2)$$
- The finite-grid search is checked against the [nominal manual reference](validation/snm_manual_check/results.md). Curve integrity matters for distorted or nonmonotonic lobes.

### Reference-normalized features
The first five ratios use a separately simulated healthy reference at the matching corner. The last three combine measurements within the target row.
1. `iddq_norm` $= I_{\text{DDQ, meas}} / I_{\text{DDQ, ref}}$
2. `t_write_norm` $= t_{\text{write, meas}} / t_{\text{write, ref}}$
3. `dv_bl_norm` $= \Delta V_{\text{BL, meas}} / \Delta V_{\text{BL, ref}}$
4. `snm_hold_norm` $= \text{HSNM}_{\text{meas}} / \text{HSNM}_{\text{ref}}$
5. `snm_read_norm` $= \text{RSNM}_{\text{meas}} / \text{RSNM}_{\text{ref}}$
6. `read_write_delay_ratio` $= t_{\text{sense}} / t_{\text{write}}$
7. `snm_ratio` $= \text{RSNM} / \text{HSNM}$
8. `dv_bl_vdd_ratio` $= \Delta V_{\text{BL}} / V_{\text{DD}}$

For the 5 kΩ opens, Table II reports lower write-delay and bitline-swing CVs after normalization. Mean normalized write delay is 1.311 for M5 and 1.382 for M6.

---


## 7. Manuscript benchmark

Each LOCO fold trains on 44 corners and tests on one corner. The paper compares Random Forest, HistGradientBoosting, Decision Tree, and Logistic Regression. The [Table III results](reports/final_results/table3_loco_benchmark.csv) use the 1,350 rows and the paper's two 12-input feature sets.

For Random Forest, pooled accuracy rises from 75.11% to 99.33% and worst-corner accuracy from 50.00% to 90.00%. On the 1,327 rows marked converged, the raw and normalized models score 75.28% and 99.70%. Omitting the entire corner containing the flagged healthy read-SNM zero gives 99.32% for the normalized model over 44 corners.

## 8. Development and manuscript artifacts

The [mid-semester presentation guide](reports/BTP_Mid_Semester_Presentation_Guide.md) and [historical 15-corner audit](reports/audit/final_comparison_table.csv) document the earlier milestone. The manuscript and [reproduction script](scripts/reproduce_manuscript_checks.py) document the 45-corner benchmark and its sensitivity checks. The additional development classifiers and nominal split belong to [the development pipeline](src/ml_pipeline.py).

---

## 9. Manuscript figures and references

- [Fig. 1](6T_SRAM_LTSPICE.jpg) shows the healthy 6T cell.
- [Fig. 2](snm_butterfly_diagram.png) shows the nominal healthy hold butterfly curve.
- [Fig. 3](fig1_feature_dispersion_comparison.png) compares CVs for the 5 kΩ opens on M5 and M6.
- [Fig. 5](fig3_rf_pvt_corner_heatmap.png) shows the 45 held-out corners.
- The [manuscript](Research_paper.tex) lists the full bibliography and includes the other evaluation figures.
