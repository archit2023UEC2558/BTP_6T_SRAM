# BTP Mid-Semester Presentation: Slide-by-Slide Script & Defense Guide

> This guide accompanies the historical mid-semester presentation and its 15-corner results. See the [manuscript](../Research_paper.tex) for the 45-corner evaluation.

**Presentation file.** [BTP_Mid_Semester_Presentation.pptx](BTP_Mid_Semester_Presentation.pptx).

**Target Format:** 16:9 Widescreen | Minimum Font Size: 24 pt | Bullet form | High-Res Graphics Embedded  
**Guideline Adherence:** Strict compliance with B.Tech Mid-Semester Project Progress Presentation Format (15 Slides)

---

## Slide-by-Slide Overview & Speaker Defense Notes

### Slide 1 — Title Slide
- **Title:** Automated SRAM Fault Diagnosis Using Physics-Normalized Feature Invariants Across PVT Corners
- **Subtitle:** Mid-Semester B.Tech Project Progress Presentation | 180 nm Level-1 MOS Models
- **Contents:**
  * Group Members: Ayush Raj (Roll No: 210102025) & Project Team
  * Project Supervisor: Faculty Advisor / Project Guide
  * Department: Department of Electronics & Communication Engineering
  * Session: Final Year B.Tech Project (BTP) — 2026
- **Speaker Delivery Tip:** Introduce the project title clearly. State that the focus is on bridging circuit physics and machine learning for reliable nanoscale semiconductor test.

---

### Slide 2 — Introduction
- **Main Idea:** Why SRAM reliability matters and why existing test methods fall short.
- **Bullets:**
  * **Problem Background:** SRAM caches occupy >70% of modern SoC area, dominating chip yield and field reliability in nanoscale nodes.
  * **Core Motivation:** Process, Voltage, and Temperature (PVT) variations cause severe operational drift, confounding fault detection.
  * **Problem Being Addressed:** Traditional test methods cannot classify latent defects, while raw ML models fail when operating corners shift.
  * **Relevance & Impact:** Essential for zero-defect mission-critical systems in automotive, aerospace, and high-reliability edge computing.
- **Anticipated Viva Question:** *Why is PVT variation a problem for machine learning in circuit testing?*
  * **Answer:** Electrical parameters like leakage current and gate delay scale non-linearly with temperature and supply voltage. A model trained only on nominal $27^\circ\text{C}, 1.0\text{V}$ data sees an out-of-distribution domain shift at $-40^\circ\text{C}$ or $125^\circ\text{C}$, causing classification collapse.

---

### Slide 3 — Objectives
- **Main Idea:** Clear statement of objectives and deliverables.
- **Bullets:**
  * **Automated Simulation Pipeline:** Develop a programmatic SPICE harness for defect injection across multi-dimensional PVT corners.
  * **Static & Dynamic Metric Extraction:** Extract butterfly Static Noise Margins (SNM), write delays, bitline discharge, and $I_{\text{DDQ}}$ currents.
  * **Reference normalization.** Compute ratios to a simulated healthy cell at the same operating point.
  * **Cross-corner evaluation and explainability.** Measure held-out corner accuracy and describe feature attributions using Tree SHAP.

---

### Slide 4 — Related Theory: 6T SRAM Cell & Static Noise Margin
- **Main Idea:** Circuit architecture and the mathematical definition of SNM.
- **Visual Embedded:** [`reports/figures/snm_butterfly_diagram.png`](../reports/figures/snm_butterfly_diagram.png)
- **Bullets:**
  * **6T SRAM Core Structure:** Cross-coupled CMOS inverter latch ($M_1$–$M_4$) with NMOS access transistors ($M_5, M_6$).
  * **Operational Stability:** Hold retention, non-destructive read stability, and successful write flipping.
  * **Mathematical SNM Definition:** Side length of the maximum square inscribed inside the butterfly curve: $\text{SNM} = \min(\text{Lobe}_1, \text{Lobe}_2)$.
  * **Read SNM Degradation:** Voltage divider bump on internal '0' node severely compresses the read stability margin.
- **Anticipated Viva Question:** *Why did you use inscribed square optimization instead of the 45-degree curve rotation shortcut?*
  * **Answer:** With rail-to-rail CMOS VTCs where $|slope| > 1$, rotating the axes creates a multi-valued curve where taking maximum separation overestimates the actual stability square. Direct square fitting is mathematically rigorous for both symmetric and asymmetric faulty cells.

---

### Slide 5 — Related Theory: Physical Fault Taxonomy & Toolchain
- **Main Idea:** Resistive defects and threshold shifts modeled in the benchmark netlist.
- **Visual Embedded:** [`reports/figures/sram_architecture_faults.png`](../reports/figures/sram_architecture_faults.png)
- **Bullets:**
  * **Class 1, resistive open.** A $200\,\Omega$ to $10\,\text{k}\Omega$ series resistance on either access path.
  * **Class 2, resistive bridge.** A $500\,\Omega$ to $10\,\text{k}\Omega$ shunt between $Q$ and $Q_b$.
  * **Class 3, storage threshold drift.** Six levels increase $|V_{\mathrm{TO}}|$ from 5% to 30% on the storage transistors.
  * **Class 4, access threshold drift.** The same six increases in $|V_{\mathrm{TO}}|$ on the access transistors.

---

### Slide 6 — Literature Survey: State of the Art & Existing Approaches
- **Main Idea:** Seminal foundations in memory testing.
- **Bullets:**
  * **Seevinck et al. (IEEE JSSC 1987):** Established analytical butterfly curve coordinate rotation for Static Noise Margin extraction.
  * **Marchal et al. (IEEE TVLSI 2008):** Classified dynamic fault behaviors in deep-submicron SRAMs using traditional March tests.
  * **Current ML Testing Works:** Apply standard classifiers (SVM, Random Forest) directly on raw digital/analog sensor readings.
  * **Key Findings:** Existing functional tests detect catastrophic failures but fail to identify latent analog degradations.

---

### Slide 7 — Literature Survey: Research Gaps & Proposed Solution
- **Main Idea:** Where prior work fails and what this BTP introduces.
- **Bullets:**
  * **Static Nominal Assumption:** Prior ML models train and test strictly at nominal conditions ($27^\circ\text{C}, 1.0\,\text{V}$), ignoring thermal and supply drift.
  * **Cross-PVT Domain Collapse:** Raw features shift up to $400\%$ across corners, causing standard ML accuracy to drop severely from $100\%$ to $<75\%$.
  * **False Overconfidence:** Random train/test splits hide out-of-distribution failure, leading to unreliable deployed diagnosis.
  * **Our proposed solution.** Use a matched simulated healthy reference and measure performance at held-out operating points.

---

### Slide 8 — Proposed Methodology: End-to-End System Workflow
- **Main Idea:** Architectural block diagram and step-by-step pipeline.
- **Visual Embedded:** [`reports/figures/methodology_flowchart.png`](../reports/figures/methodology_flowchart.png)
- **Bullets:**
  * **1. Automated Netlist Builder:** Constructs 180 nm Level-1 SPICE netlists with four injected fault classes.
  * **2. Multi-Corner Simulation:** Runs the 15 voltage–temperature combinations at the nominal process setting using LTspice ($0.9\text{V}$–$1.1\text{V}$, $-40^\circ\text{C}$–$125^\circ\text{C}$).
  * **3. Feature Extraction:** Computes dynamic metrics & inscribed-square SNM margins.
  * **4. Reference normalization.** Forms $X_{\text{meas}} / X_{\text{ref}}$ using a healthy cell simulated at the same operating point.
  * **5. Cross-PVT Benchmarking:** Evaluates generalization using Leave-One-Corner-Out across 4 ML models.

---

### Slide 9 — Work Completed: Multi-Corner Simulation & Data Matrix
- **Main Idea:** Dataset scale, integrity, and simulation reliability.
- **Bullets:**
  * **Complete Simulation Matrix:** Successfully simulated 450 instances across 15 distinct PVT corners and 5 physical fault classes.
  * **Zero Data Leakage / Corruption:** Clean dataset generated with 0 missing values, 0 NaNs, and 0 infinite values across all features.
  * **Convergence Logging:** Parsed and audited 295 simulator convergence warnings to ensure physical simulation validity.
  * **Inscribed Square Accuracy:** Validated butterfly SNM extraction against reference golden curves with $<1.0\,\text{mV}$ mathematical error.

---

### Slide 10 — Work Completed: Feature Invariance & PVT Drift Mitigation
- **Main Idea:** Visual and statistical proof that physics-normalization works.
- **Visual Embedded:** [`reports/figures/fig1_feature_dispersion_comparison.png`](../reports/figures/fig1_feature_dispersion_comparison.png)
- **Bullets:**
  * **Raw Parameter Instability:** Raw delays and $I_{\text{DDQ}}$ currents exhibit $>60\%$ coefficient of variation across temperatures.
  * **Invariant Collapse:** Normalized features collapse environmental dispersion to $<5\%$ for healthy cells.
  * **Preserved Separability:** Fault signatures remain distinctly clustered regardless of operating corner shifts.
  * **Demonstrated Proof:** Direct visual evidence shown in adjacent distribution plot across 15 corners.

---

### Slide 11 — Work Completed: Cross-PVT LOCO Diagnostic Performance
- **Main Idea:** Empirical benchmark proving high cross-corner generalization.
- **Historical comparison figure.** [Original and cleaned 15-corner results](audit/final_comparison_figure.png)
- **Bullets:**
  * **Random Forest and HistGB.** Both reached **98.89%** mean LOCO accuracy with normalized features. The raw results were **82.67%** and **82.89%**, respectively. [Archived summary](audit/loco_summary_original_450.csv)
  * **Linear models.** Linear SVM and Logistic Regression improved by 11.55 and 6.22 percentage points with normalized features.
  * **Non-ML Baseline Failure:** Standard $3\sigma$ rule detector degrades heavily across corners, confirming need for ML.
  * **Sensitivity audit.** The normalized Random Forest reached 100.00% after excluding 22 rows with missing waveform output. [Historical comparison](audit/final_comparison_table.csv)

---

### Slide 12 — Project Progress Against Original Plan
- **Main Idea:** Timeline tracking showing 100% completion of planned milestones.
- **Bullets:**
  * **Phase 1: Circuit & Simulation (Weeks 1–4):** 180 nm Level-1 netlists, sizing, and automated fault injection operational.
  * **Phase 2: Dataset & Invariant Features (Weeks 5–7):** 100% Completed | 450 simulation matrix and normalization pipeline fully verified.
  * **Phase 3: Model Training & Evaluation (Weeks 7–8):** 100% Completed | 4 classical ML models evaluated under Standard Split and LOCO.
  * **Phase 4: Explainability & Checkpoints (Week 9):** 100% Completed | Tree SHAP feature attribution & all 7 Section 8 checkpoints passed.
  * **Deviations from Original Plan:** None. Ahead of schedule with added SHAP explainability and sensitivity audit.

---

### Slide 13 — Work to be Completed by End Semester & Proposed Timeline
- **Main Idea:** Future work scope leading to final submission.
- **Bullets:**
  * **Formal Research Manuscript:** Finalize full academic paper (IEEE transactions/conference format) synthesizing results.
  * **Process Corner & Statistical Variation Modeling:** Extend the framework to evaluate inter-die process corners (FF, SS) and Monte Carlo variations.
  * **Extended Fault Taxonomy:** Explore multi-cell coupling defects and adjacent bitline bridging faults.
  * **Target Completion Schedule:** Weeks 10–12: Manuscript writing & peer review | Weeks 13–14: Final thesis & oral defense.

---

### Slide 14 — Conclusion & Current Project Status
- **Main Idea:** Executive takeaway summarizing the novelty and readiness.
- **Bullets:**
  * **Key result.** Reference-normalized features improved classification across the simulated operating grid.
  * **Historical LOCO results.** The normalized Random Forest reached 98.89% across 15 folds of the original dataset and 100.00% on the 428-row subset.
  * **Validation artifacts.** The current checkpoint report records stored-data and nominal deck-assembly checks for the 45-corner benchmark. [Checkpoint report](validation_checkpoints_report.md)
  * **Current Project Status:** Engineering, modeling, and validation 100% complete; paper manuscript in progress.

---

### Slide 15 — References & Technical Documentation
- **Main Idea:** Peer-reviewed academic foundations.
- **Bullets:**
  * **[1] J. Seevinck et al.:** "Static-Noise Margin Analysis of MOS SRAM Cells," *IEEE J. Solid-State Circuits*, vol. 22, no. 5, 1987.
  * **[2] P. Marchal et al.:** "SRAM Dynamic Fault Modeling and Testing in Deep-Submicron Technologies," *IEEE TVLSI*, 2008.
  * **[3] Predictive Technology Model:** "PTM 45nm Low Power CMOS Model Cards," Arizona State University (Nanoscale PTM).
  * **[4] S. Lundberg & S. Lee:** "A Unified Approach to Interpreting Model Predictions (SHAP)," *NeurIPS*, 2017.
  * **[5] Project Repository:** Automated SRAM Fault Diagnosis Pipeline, 2026. (https://github.com/AyushRaj2506/BTP-SRAM).
