"""Recompute Table II and the manuscript's diagnostic checks from the saved CSV.

    python scripts/reproduce_manuscript_checks.py
    python scripts/reproduce_manuscript_checks.py --rf --shap

The first command needs NumPy and pandas. The optional model checks also use
scikit-learn and, for Tree SHAP, shap. Results are printed to standard output.
"""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path(__file__).resolve().parents[1] / "data" / "sram_fault_dataset.csv"
FLAGGED_READ_CORNER = "TT_C_0C_0P9V"
BASE_FEATURE_PAIRS = [
    ("i_ddq_uA", "iddq_norm"),
    ("t_write_ps", "t_write_norm"),
    ("dv_bl_strobe_mV", "dv_bl_norm"),
    ("snm_hold_v", "snm_hold_norm"),
    ("snm_read_v", "snm_read_norm"),
]
RAW_BASE = [raw for raw, _ in BASE_FEATURE_PAIRS]
FLAGS = ["hold_pass", "write_pass", "read_stable_pass", "is_bistable"]
RAW_INPUTS = RAW_BASE + ["i_write_peak_mA", "t_sense_ps", "v_bump_mV"] + FLAGS
RATIO_INPUTS = [ratio for _, ratio in BASE_FEATURE_PAIRS] + [
    "read_write_delay_ratio", "snm_ratio", "dv_bl_vdd_ratio",
] + FLAGS
FOREST_SETTINGS = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": 1}
RATIO_FLOOR = 1e-12


def coefficient_of_variation(values: pd.Series) -> float:
    """Return the sample CV as a percentage."""
    mean = values.mean()
    return 100 * values.std(ddof=1) / mean if mean != 0 else float("nan")


def print_table_ii(data: pd.DataFrame) -> None:
    healthy = data.loc[data["fault_class"].eq(0)]
    converged = data["converged"].eq(1)
    fixed_opens = data.loc[converged & data["fault_class"].eq(1) & data["severity_value"].eq(5000)]
    groups = {
        "Healthy (all)": healthy,
        "Healthy (converged)": healthy.loc[healthy["converged"].eq(1)],
        "5 kOhm open (M5)": fixed_opens.loc[fixed_opens["fault_target"].eq("M5")],
        "5 kOhm open (M6)": fixed_opens.loc[fixed_opens["fault_target"].eq("M6")],
    }
    records = []
    for name, group in groups.items():
        if len(group) < 2 or not group["corner_id"].is_unique:
            raise ValueError(f"{name} needs one observation per corner across at least two corners")
        for raw, normalized in BASE_FEATURE_PAIRS:
            records.append((
                name, len(group), normalized, f"{group[normalized].mean():.3f}",
                f"{coefficient_of_variation(group[raw]):.2f}",
                f"{coefficient_of_variation(group[normalized]):.2f}",
            ))
    columns = ["Group", "Corners", "Feature", "Normalized mean", "Raw CV (%)", "Normalized CV (%)"]
    print("\nTable II. Sample CVs use ddof=1. Both CVs use the same rows in each group.")
    print(pd.DataFrame(records, columns=columns).to_string(index=False))

    flagged_corner = data.loc[data["corner_id"].eq(FLAGGED_READ_CORNER)]
    flagged_healthy = flagged_corner.loc[flagged_corner["fault_class"].eq(0),
                                         ["converged", "snm_read_v", "snm_read_norm"]]
    print(f"\nSection V-C. Stored healthy read SNM at {FLAGGED_READ_CORNER}:")
    print(flagged_healthy.to_string(index=False))
    positive = flagged_corner.loc[flagged_corner["snm_read_norm"].gt(0)]
    reference_mv = 1000 * positive["snm_read_v"] / positive["snm_read_norm"]
    print("Reference denominators recovered from positive rows (mV): "
          + ", ".join(f"{value:.3f}" for value in reference_mv))

    print(f"\nRead SNM after excluding {FLAGGED_READ_CORNER}:")
    for name, group in groups.items():
        values = group.loc[group["corner_id"].ne(FLAGGED_READ_CORNER), "snm_read_norm"]
        print(f"{name}: {len(values)} corners, mean {values.mean():.3f}, "
              f"CV {coefficient_of_variation(values):.2f}%")

    fault_levels = data.loc[data["fault_class"].ne(0)].groupby("fault_class")["severity_value"].nunique()
    print(f"\nDistinct sampled parameter levels by fault class: {fault_levels.to_dict()}")
    nominal = healthy.loc[healthy["corner_id"].eq("TT_C_27C_1P0V"), ["snm_read_v", "v_bump_mV"]]
    print("Nominal healthy read measurements (V and mV):")
    print(nominal.to_string(index=False))


def report_loco(
    name: str, rows: pd.DataFrame, inputs: np.ndarray, *, max_features: int | str = "sqrt"
) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score

    corner_ids = rows["corner_id"].to_numpy()
    actual = rows["fault_class"].to_numpy()
    predicted = np.empty_like(actual)
    corners = np.unique(corner_ids)
    for corner in corners:
        held_out = corner_ids == corner
        forest = RandomForestClassifier(max_features=max_features, **FOREST_SETTINGS)
        forest.fit(inputs[~held_out], actual[~held_out])
        predicted[held_out] = forest.predict(inputs[held_out])

    correct = predicted == actual
    accuracy = correct.mean()
    balanced = balanced_accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro")
    corner_accuracy = pd.Series(correct).groupby(corner_ids).mean()
    print(f"{name}: {correct.sum()}/{len(actual)}, accuracy {accuracy:.2%}, "
          f"balanced accuracy {balanced:.2%}, macro-F1 {macro_f1:.2%}, "
          f"worst corner {corner_accuracy.min():.2%}, "
          f"perfect corners {(corner_accuracy == 1).sum()}/{len(corners)}", flush=True)
    return float(accuracy)


def print_rf_controls(data: pd.DataFrame) -> None:
    import sklearn

    # Apply the row filter before splitting, so excluded rows never enter a fold.
    converged = data.loc[data["converged"].eq(1)]
    print(f"\nConverged-only LOCO (scikit-learn {sklearn.__version__}). "
          f"{len(data) - len(converged)} rows excluded before splitting.", flush=True)
    report_loco("Raw inputs and flags", converged, converged[RAW_INPUTS].to_numpy())
    report_loco("Ratios and flags", converged, converged[RATIO_INPUTS].to_numpy())

    # Table IV uses the same stored healthy row for all transformations and the
    # unnormalized control, including the known reference at the held-out corner.
    reference = data.loc[data["fault_class"].eq(0)].set_index("corner_id")
    target = data[RAW_BASE].to_numpy()
    same_corner = reference.loc[data["corner_id"], RAW_BASE].to_numpy()
    denominator = np.where(np.abs(same_corner) > RATIO_FLOOR, same_corner, RATIO_FLOOR)
    ratios = np.maximum(target / denominator, RATIO_FLOOR)
    raw_with_reference = np.column_stack((target, same_corner))
    inputs = {
        "Raw (5)": target,
        "Raw + reference (10)": raw_with_reference,
        "Difference (5)": target - same_corner,
        "Log-ratio (5)": np.log(ratios),
        "Ratio (5)": ratios,
    }
    print("\nTable IV matched controls (max_features=2).", flush=True)
    scores = {}
    for name, features in inputs.items():
        scores[name] = report_loco(name, data, features, max_features=2)
    print(f"Gain from adding the raw reference: "
          f"{100 * (scores['Raw + reference (10)'] - scores['Raw (5)']):.2f} percentage points.")
    print(f"Gain from forming ratios over raw + reference: "
          f"{100 * (scores['Ratio (5)'] - scores['Raw + reference (10)']):.2f} percentage points.")
    report_loco("Raw + reference (10), square-root feature sampling", data, raw_with_reference)


def print_read_snm_sensitivities(data: pd.DataFrame) -> None:
    print("\nSection VI-C. Read-SNM sensitivity checks.", flush=True)
    corner_excluded = data.loc[data["corner_id"].ne(FLAGGED_READ_CORNER)]
    report_loco(f"Ratios and flags, excluding {FLAGGED_READ_CORNER} before splitting",
                corner_excluded, corner_excluded[RATIO_INPUTS].to_numpy())
    read_snm_inputs = {"snm_read_norm", "snm_ratio", "is_bistable"}
    reduced_inputs = [feature for feature in RATIO_INPUTS if feature not in read_snm_inputs]
    report_loco("Ratios and flags, omitting read-SNM-derived inputs",
                data, data[reduced_inputs].to_numpy())


def print_shap_rankings(data: pd.DataFrame) -> None:
    import shap
    from sklearn.ensemble import RandomForestClassifier

    subsets = {
        "Full data": data,
        f"Excluding {FLAGGED_READ_CORNER}": data.loc[data["corner_id"].ne(FLAGGED_READ_CORNER)],
    }
    for name, rows in subsets.items():
        print(f"\nTree SHAP (shap {shap.__version__}). {name}, {len(rows)} rows.", flush=True)
        features = rows[RATIO_INPUTS]
        forest = RandomForestClassifier(**FOREST_SETTINGS).fit(features, rows["fault_class"])
        values = shap.TreeExplainer(forest).shap_values(features)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        by_class = np.abs(values).mean(axis=0)
        ranking = pd.DataFrame(by_class, index=RATIO_INPUTS,
                               columns=[f"Class {c}" for c in forest.classes_])
        ranking.insert(0, "Global", by_class.mean(axis=1))
        print(ranking.sort_values("Global", ascending=False).to_string(float_format="{:.4f}".format))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the SRAM manuscript's Table II and model checks.")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="Benchmark CSV path")
    parser.add_argument("--rf", action="store_true", help="Refit and score the Random Forest LOCO controls")
    parser.add_argument("--shap", action="store_true", help="Refit the Random Forest and calculate Tree SHAP rankings")
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    if data["sample_id"].duplicated().any() or data["corner_id"].isna().any():
        raise ValueError("Expected unique sample IDs and a corner ID on each benchmark row")
    print(f"Dataset: {args.dataset.resolve()} ({len(data)} rows)")
    print(f"SHA-256: {sha256(args.dataset.read_bytes()).hexdigest()}")
    print_table_ii(data)
    if args.rf:
        print_rf_controls(data)
        print_read_snm_sensitivities(data)
    if args.shap:
        print_shap_rankings(data)


if __name__ == "__main__":
    main()
