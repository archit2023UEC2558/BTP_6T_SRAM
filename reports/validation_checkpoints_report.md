# Validation checkpoints for the 45-corner benchmark

Regenerate this report with the [checkpoint script](../scripts/verify_checkpoints.py).
The checks use saved data and assemble nominal testbenches in a temporary directory.

| Checkpoint | Outcome |
|---|---|
| Nominal healthy observation | PASS |
| Synthetic SNM extraction | PASS |
| Fault deck assembly | PASS |
| Development baseline sanity | PASS |
| Dataset completeness | PASS |
| Sample ID overlap | PASS |
| Stored paper result consistency | PASS |

## Nominal healthy observation

- **Source** — Stored observation SRAM_F0_S0_27C_1P0V_0211 at TT_C_27C_1P0V
- **Hold, read stability, write, bistability** — 1, 1, 1, 1
- **Sampled standby current at 20 ns** — 4.2 nA
- **Mean bitline differential at 2.15 ns** — 224.8 mV
- **Mean sense delay** — 60.0 ps
- **Mean delay across the two write directions** — 60.2 ps

## Synthetic SNM extraction

- **Synthetic lobe sides** — 400.00 mV, 400.00 mV
- **Expected side length** — 400.00 mV per lobe
- **Maximum deviation** — 0.000000 mV (tolerance 1 mV)
- **Nominal LTspice measurements** — See [the manual SNM check](../validation/snm_manual_check/results.md)

## Fault deck assembly

- **Nominal configurations assembled** — 30
- **Operation decks assembled** — 210
- **Scope** — Netlist generation and composer node validation

## Development baseline sanity

- **Healthy fit and screening rows** — 45
- **In-sample false-positive rate** — 2.22% (threshold 15%)
- **Thresholds fitted** — write_time_max

## Dataset completeness

- **Dataset rows, columns, corners** — 1,350, 34, 45
- **Missing or infinite entries** — 0
- **Duplicate sample IDs** — 0
- **Rows marked converged** — 1,327
- **Rows marked nonconverged** — 23
- **Interpretation** — Completeness and row-status checks on the stored data. The convergence flag can retain a rejected read-SNM sweep.

## Sample ID overlap

- **Nominal split train and test rows** — 24, 6
- **Nominal split sample ID overlap** — 0
- **LOCO folds checked** — 45
- **Total LOCO sample ID overlap** — 0

## Stored paper result consistency

- **Source** — [Stored Table III results](final_results/table3_loco_benchmark.csv)
- **Random Forest** — Raw 75.11%, normalized 99.33%, gain +24.22 pp, worst corner 50.00% to 90.00%
- **HistGradientBoosting** — Raw 77.56%, normalized 99.41%, gain +21.85 pp, worst corner 40.00% to 86.67%
- **Decision Tree** — Raw 73.56%, normalized 98.44%, gain +24.89 pp, worst corner 36.67% to 63.33%
- **Logistic Regression** — Raw 80.59%, normalized 86.89%, gain +6.30 pp, worst corner 60.00% to 60.00%

Use the [manuscript reproduction script](../scripts/reproduce_manuscript_checks.py)
to recompute dispersion, Random Forest sensitivity results and SHAP values.
