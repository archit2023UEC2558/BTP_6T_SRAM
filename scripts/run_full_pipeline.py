"""
scripts/run_full_pipeline.py -- Full End-to-End Orchestrator for SRAM Fault Diagnosis.

Executes the complete pipeline:
  Step 1: 45-Corner Dataset Generation (3 Process x 15 V-T = 45 corners, 1,350 rows)
  Step 2: Machine Learning Evaluation (Standard Split + LOCO across 45 corners)
  Step 3: Cross-PVT / Process Statistical Analysis & Publication Figures
  Step 4: SHAP Explainability Analysis
  Step 5: Overleaf figure copy (root flat structure)
"""

import sys
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from generate_dataset import generate_sram_dataset
from src import ml_pipeline
from src import cross_pvt_analysis
import run_shap_analysis


def main():
    t_start = time.time()
    print("=====================================================================")
    print("SRAM FAULT DIAGNOSIS - FULL 45-CORNER PVT & PROCESS PIPELINE")
    print("=====================================================================")

    # -----------------------------------------------------------------------
    # Step 1: Generate Expanded Dataset
    # -----------------------------------------------------------------------
    print("\n>>> STEP 1: GENERATING 45-CORNER DATASET (1,350 SAMPLES)...")
    df_data = generate_sram_dataset(max_workers=8, prune_raw=True)
    print(f"Dataset generation completed: {len(df_data)} rows generated.")

    # -----------------------------------------------------------------------
    # Step 2: Run Machine Learning Pipeline
    # -----------------------------------------------------------------------
    print("\n>>> STEP 2: RUNNING ML EVALUATION (STANDARD SPLIT & LOCO)...")
    dataset_path = ROOT / "data" / "sram_fault_dataset.csv"
    df = pd.read_csv(dataset_path)

    feature_sets = {
        "raw": ml_pipeline.RAW_FEATURES,
        "invariant": ml_pipeline.INVARIANT_FEATURES,
    }

    ml_pipeline.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ml_pipeline.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    std_results, std_models = ml_pipeline.run_standard_split_protocol(df, feature_sets)
    loco_results, loco_models = ml_pipeline.run_leave_one_corner_out_protocol(df, feature_sets)

    all_results = std_results + loco_results
    df_res = pd.DataFrame(all_results)
    out_table = ml_pipeline.REPORTS_DIR / "comparison_table.csv"
    df_res.to_csv(out_table, index=False)
    print(f"Saved comparison table to {out_table} ({len(df_res)} records)")

    import joblib
    for name, model in {**std_models, **loco_models}.items():
        pkl_path = ml_pipeline.MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, pkl_path)

    # -----------------------------------------------------------------------
    # Step 3: Run Cross-PVT Analysis & Figures
    # -----------------------------------------------------------------------
    print("\n>>> STEP 3: COMPUTING CROSS-PVT & PROCESS STATS AND FIGURES...")
    cross_pvt_analysis.analyze_feature_spread(df)
    cross_pvt_analysis.plot_loco_comparison(out_table)

    # -----------------------------------------------------------------------
    # Step 4: Run SHAP Analysis
    # -----------------------------------------------------------------------
    print("\n>>> STEP 4: RUNNING SHAP EXPLAINABILITY ANALYSIS...")
    run_shap_analysis.main()

    # -----------------------------------------------------------------------
    # Step 5: Copy figures to project root for Overleaf flat upload
    # -----------------------------------------------------------------------
    print("\n>>> STEP 5: SYNCING FIGURES TO ROOT FOR OVERLEAF...")
    fig_dir = ROOT / "reports" / "figures"
    copied = 0
    for img in fig_dir.glob("*.png"):
        dest = ROOT / img.name
        shutil.copy2(img, dest)
        copied += 1
    print(f"Synced {copied} figures to root.")

    elapsed = time.time() - t_start
    print(f"\n=====================================================================")
    print(f"PIPELINE COMPLETE IN {elapsed/60:.2f} MINUTES!")
    print(f"=====================================================================")


if __name__ == "__main__":
    main()
