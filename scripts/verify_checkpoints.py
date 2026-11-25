"""Regenerate the validation summary for the paper's 45-corner benchmark."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_detector import BaselineRuleDetector
from src.generate_dataset import HEALTHY_CORE_PATH, OPERATIONS, create_fault_cell, get_default_fault_specs
from src.ml_pipeline import load_dataset
from src.snm_extraction import lobe_sides
from src.testbench_composer import compose_testbench
from src.utils import load_cell_core

REPORT_PATH = ROOT / "reports" / "validation_checkpoints_report.md"
BENCHMARK_PATH = ROOT / "reports" / "final_results" / "table3_loco_benchmark.csv"
NOMINAL_CORNER = "TT_C_27C_1P0V"
CORNERS = 45
CONFIGURATIONS_PER_CORNER = 30
PAPER_MODELS = ("Random Forest", "HistGradientBoosting", "Decision Tree", "Logistic Regression")


def check_healthy_cell(data: pd.DataFrame):
    nominal = data.loc[data["corner_id"].eq(NOMINAL_CORNER) & data["fault_class"].eq(0)]
    if len(nominal) != 1:
        return False, {"Nominal healthy rows": len(nominal)}
    row = nominal.iloc[0]
    checks = ["hold_pass", "read_stable_pass", "write_pass", "is_bistable"]
    return bool(row[checks].eq(1).all()), {
        "Source": f"Stored observation {row['sample_id']} at {NOMINAL_CORNER}",
        "Hold, read stability, write, bistability": ", ".join(str(int(row[flag])) for flag in checks),
        "Sampled standby current at 20 ns": f"{1000 * row['i_ddq_uA']:.1f} nA",
        "Mean bitline differential at 2.15 ns": f"{row['dv_bl_strobe_mV']:.1f} mV",
        "Mean sense delay": f"{row['t_sense_ps']:.1f} ps",
        "Mean delay across the two write directions": f"{row['t_write_ps']:.1f} ps",
    }


def check_snm_extraction():
    """Exercise the extractor against a symmetric piecewise-linear VTC pair."""
    sweep = np.linspace(0.0, 1.0, 1001)
    curve = np.interp(sweep, [0.0, 0.4, 0.6, 1.0], [1.0, 1.0, 0.0, 0.0])
    # Each rail-to-transition interval supports a 0.4 V square. The sloped
    # segment bounds candidate squares beyond that side length.
    lobes = lobe_sides(sweep, curve, curve, sweep, vdd=1.0)
    expected = 0.4
    max_error = max(abs(lobe - expected) for lobe in lobes)
    return max_error < 0.001, {
        "Synthetic lobe sides": ", ".join(f"{1000 * lobe:.2f} mV" for lobe in lobes),
        "Expected side length": f"{1000 * expected:.2f} mV per lobe",
        "Maximum deviation": f"{1000 * max_error:.6f} mV (tolerance 1 mV)",
        "Nominal LTspice measurements": "See [the manual SNM check](../validation/snm_manual_check/results.md)",
    }


def check_fault_deck_assembly():
    """Build the nominal healthy and fault decks using the production composer."""
    core = load_cell_core(str(HEALTHY_CORE_PATH))
    specs = get_default_fault_specs()
    with TemporaryDirectory(prefix="sram-checkpoints-") as directory:
        work = Path(directory)
        for index, spec in enumerate(specs):
            core_path = work / f"core_{index}.net"
            create_fault_cell(spec, core, str(core_path))
            for operation in OPERATIONS:
                deck = work / f"case_{index}_{operation}.net"
                compose_testbench(str(core_path), operation, str(deck), vdd=1.0, temperature_c=27)
    return len(specs) == CONFIGURATIONS_PER_CORNER, {
        "Nominal configurations assembled": len(specs),
        "Operation decks assembled": len(specs) * len(OPERATIONS),
        "Scope": "Netlist generation and composer node validation",
    }


def check_baseline_sanity(data: pd.DataFrame):
    healthy = data.loc[data["fault_class"].eq(0)]
    if healthy.empty:
        return False, {"Healthy rows": 0}
    detector = BaselineRuleDetector(n_sigma=3.0).fit(healthy)
    false_alarm_rate = float(np.mean(detector.predict(healthy) == 1))
    return false_alarm_rate <= 0.15, {
        "Healthy fit and screening rows": len(healthy),
        "In-sample false-positive rate": f"{false_alarm_rate:.2%} (threshold 15%)",
        "Thresholds fitted": ", ".join(detector.thresholds_),
    }


def check_dataset_integrity(data: pd.DataFrame):
    counts = data.groupby("corner_id").size()
    null_count = int(data.isna().sum().sum())
    inf_count = int(np.isinf(data.select_dtypes(include=np.number)).sum().sum())
    duplicate_ids = int(data["sample_id"].duplicated().sum())
    converged = data["converged"].eq(1)
    passed = (
        len(counts) == CORNERS
        and counts.eq(CONFIGURATIONS_PER_CORNER).all()
        and len(data.columns) == 34
        and null_count == inf_count == duplicate_ids == 0
        and data["converged"].isin([0, 1]).all()
    )
    return passed, {
        "Dataset rows, columns, corners": f"{len(data):,}, {len(data.columns)}, {len(counts)}",
        "Missing or infinite entries": null_count + inf_count,
        "Duplicate sample IDs": duplicate_ids,
        "Rows marked converged": f"{converged.sum():,}",
        "Rows marked nonconverged": int((~converged).sum()),
        "Interpretation": "Completeness and row-status checks on the stored data. The convergence flag can retain a rejected read-SNM sweep.",
    }


def check_split_overlap(data: pd.DataFrame):
    nominal = data.loc[data["corner_id"].eq(NOMINAL_CORNER)]
    if len(nominal) != CONFIGURATIONS_PER_CORNER:
        return False, {"Nominal rows": len(nominal)}
    train, test = train_test_split(nominal, test_size=0.20, random_state=42)
    nominal_overlap = len(set(train["sample_id"]) & set(test["sample_id"]))
    corner_ids = data["corner_id"]
    overlaps = []
    for corner in corner_ids.unique():
        held_out = corner_ids.eq(corner)
        train_ids = set(data.loc[~held_out, "sample_id"])
        test_ids = set(data.loc[held_out, "sample_id"])
        overlaps.append(len(train_ids & test_ids))
    loco_overlap = sum(overlaps)
    return nominal_overlap == loco_overlap == 0, {
        "Nominal split train and test rows": f"{len(train)}, {len(test)}",
        "Nominal split sample ID overlap": nominal_overlap,
        "LOCO folds checked": len(overlaps),
        "Total LOCO sample ID overlap": loco_overlap,
    }


def check_paper_results():
    summary = pd.read_csv(BENCHMARK_PATH).set_index("Model")
    if not summary.index.is_unique or set(summary.index) != set(PAPER_MODELS):
        return False, {"Models found": ", ".join(summary.index)}
    gain = 100 * (summary["Inv_Acc"] - summary["Raw_Acc"])
    passed = (
        gain.gt(0).all()
        and np.allclose(summary["Gain_Pct"], gain)
        and summary["Perfect_Corners"].str.endswith(f"/{CORNERS}").all()
    )
    details = {"Source": "[Stored Table III results](final_results/table3_loco_benchmark.csv)"}
    for model in PAPER_MODELS:
        row = summary.loc[model]
        details[model] = (
            f"Raw {row['Raw_Acc']:.2%}, normalized {row['Inv_Acc']:.2%}, "
            f"gain {gain[model]:+.2f} pp, worst corner {row['Raw_Min']:.2%} to {row['Inv_Min']:.2%}"
        )
    return passed, details


def main() -> int:
    data = load_dataset()
    checks = [
        ("Nominal healthy observation", check_healthy_cell(data)),
        ("Synthetic SNM extraction", check_snm_extraction()),
        ("Fault deck assembly", check_fault_deck_assembly()),
        ("Development baseline sanity", check_baseline_sanity(data)),
        ("Dataset completeness", check_dataset_integrity(data)),
        ("Sample ID overlap", check_split_overlap(data)),
        ("Stored paper result consistency", check_paper_results()),
    ]
    lines = [
        "# Validation checkpoints for the 45-corner benchmark",
        "",
        "Regenerate this report with the [checkpoint script](../scripts/verify_checkpoints.py).",
        "The checks use saved data and assemble nominal testbenches in a temporary directory.",
        "",
        "| Checkpoint | Outcome |",
        "|---|---|",
    ]
    for name, (passed, _) in checks:
        result = "PASS" if passed else "FAIL"
        lines.append(f"| {name} | {result} |")
        print(f"[{result}] {name}")
    for name, (_, details) in checks:
        lines.extend(["", f"## {name}", ""])
        lines.extend(f"- **{key}** — {value}" for key, value in details.items())

    lines.extend([
        "",
        "Use the [manuscript reproduction script](../scripts/reproduce_manuscript_checks.py)",
        "to recompute dispersion, Random Forest sensitivity results and SHAP values.",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {REPORT_PATH}")
    return 0 if all(passed for _, (passed, _) in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
