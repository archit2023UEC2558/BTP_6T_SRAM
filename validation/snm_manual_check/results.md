# Task 0 Validation: SNM Manual Check Results

**Date:** 2026-09-28  
**Decks:** `circuits/reference/snm_read_healthy.net`, `circuits/reference/snm_hold_healthy.net`  
**Simulator:** LTspice (via `simulation_runner.run_simulation`)  
**Extraction Method:** Largest inscribed axis-aligned square (`snm_extraction.py`, Appendix A)  

---

## 1. Summary Results

The rounded references provide a consistency check for the nominal LTspice sweeps. The finite search grid and interpolation limit the extraction resolution.

| Mode | Lobe 1 (State 0) | Lobe 2 (State 1) | SNM | Asymmetry | Recorded reference | Difference from reference |
|---|---|---|---|---|---|---|
| **Read SNM (Healthy)** | 263.6900 mV | 263.6900 mV | **263.6900 mV** (0.2637 V) | 0.0000 mV | 263.70 mV | −0.01 mV |
| **Hold SNM (Healthy)** | 470.0579 mV | 470.0579 mV | **470.0579 mV** (0.4701 V) | 0.0000 mV | 470.00 mV | +0.06 mV |

---

## 2. Voltage Transfer Curve (VTC) Checkpoints: $Q \to Qb$

### Read Mode VTC ($V_{WL} = 1.0\text{ V}$, $V_{BL} = V_{BLB} = 1.0\text{ V}$)

| $V(Q)$ [V] | Recorded reference [V] | LTspice measured [V] | Difference [mV] |
|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 0.0 |
| 0.30 | 1.0000 | 1.0000 | 0.0 |
| 0.40 | 1.0000 | 1.0000 | 0.0 |
| 0.45 | 1.0000 | 1.0000 | 0.0 |
| 0.50 | 0.4118 | 0.4118 | 0.0 |
| 0.55 | 0.3588 | 0.3588 | 0.0 |
| 0.60 | 0.3116 | 0.3116 | 0.0 |
| 0.70 | 0.2204 | 0.2204 | 0.0 |
| 1.00 | 0.1268 | 0.1268 | 0.0 |

The tabulated read VTC points agree with the recorded reference to four decimal places.

---

### Hold Mode VTC ($V_{WL} = 0.0\text{ V}$, $V_{BL} = V_{BLB} = 1.0\text{ V}$)

| $V(Q)$ [V] | Recorded reference [V] | LTspice measured [V] | Difference [mV] |
|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 0.0 |
| 0.30 | 1.0000 | 1.0000 | 0.0 |
| 0.40 | 1.0000 | 1.0000 | 0.0 |
| 0.45 | 1.0000 | 1.0000 | 0.0 |
| 0.50 | 0.0058 | 0.0058 | 0.0 |
| 0.55 | 0.0000 | 0.0000 | 0.0 |
| 0.60 | 0.0000 | 0.0000 | 0.0 |
| 0.70 | 0.0000 | 0.0000 | 0.0 |
| 1.00 | 0.0000 | 0.0000 | 0.0 |

The tabulated hold VTC points agree with the recorded reference to four decimal places.

---

## 3. Simulation Artifacts

- **Read Deck:** `circuits/reference/snm_read_healthy.net`
  - `.raw`: `validation/snm_manual_check/raw/snm_read_healthy_9297259c.raw`
  - `.log`: `validation/snm_manual_check/logs/snm_read_healthy_9297259c.log`
  - Convergence: True, 0 fatal errors, 1001 points swept.
- **Hold Deck:** `circuits/reference/snm_hold_healthy.net`
  - `.raw`: `validation/snm_manual_check/raw/snm_hold_healthy_621e25f3.raw`
  - `.log`: `validation/snm_manual_check/logs/snm_hold_healthy_621e25f3.log`
  - Convergence: True, 0 fatal errors, 1001 points swept.
