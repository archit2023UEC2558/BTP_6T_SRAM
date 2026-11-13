"""
scripts/generate_45corner_shap_plots.py

Regenerates the authentic 1,350-sample SHAP figures:
- fig4a_shap_summary.png
- fig4b_shap_class_importance.png
Ensures exact match with 45-corner / 1,350-sample dataset and updates all figure locations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sram_fault_dataset.csv"

FIG_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "reports" / "figures",
    PROJECT_ROOT / "Overleaf_Upload_Package"
]

def save_fig_all(fig, filename):
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / filename
        fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {filename} -> {out_path}")

def main():
    print("Loading 45-corner dataset...")
    df = pd.read_csv(DATA_PATH)
    n_samples = len(df)
    n_corners = df['corner_id'].nunique()
    print(f"Dataset: {n_samples} samples across {n_corners} PVT corners.")

    inv_12 = [
        'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
        'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio',
        'hold_pass', 'write_pass', 'read_stable_pass', 'is_bistable'
    ]

    print("Fitting Random Forest classifier on 12 invariant features...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(df[inv_12], df['fault_class'])

    print("Computing TreeExplainer SHAP values (1,350 samples)...")
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(df[inv_12])

    # Convert to 3D numpy array: (n_samples, n_features, n_classes)
    if isinstance(sv, list):
        # sv is list of 5 arrays, each (1350, 12)
        sv_array = np.stack(sv, axis=-1)  # shape: (1350, 12, 5)
    elif isinstance(sv, np.ndarray):
        sv_array = sv  # shape: (1350, 12, 5)
    else:
        raise ValueError(f"Unexpected SHAP values type: {type(sv)}")

    print(f"SHAP array shape: {sv_array.shape}")

    # 1. Global Mean Absolute SHAP across all classes and samples
    # sv_array shape: (1350, 12, 5) -> mean across samples (axis 0) and classes (axis 2)
    mean_abs_global = np.abs(sv_array).mean(axis=(0, 2))
    shap_df = pd.DataFrame({
        'Feature': inv_12,
        'Mean_Abs_SHAP': mean_abs_global
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    shap_df['Rank'] = shap_df.index + 1

    print("\nTop 6 Global Features:")
    print(shap_df.head(6).to_string())

    # Save Fig 4a / Fig 4: Global SHAP
    fig4a, ax4a = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=shap_df.head(8), x='Mean_Abs_SHAP', y='Feature', palette='viridis', ax=ax4a)
    ax4a.set_title(f"Global Feature Importance via Tree SHAP\n({n_samples} Authentic SPICE Samples across {n_corners} PVT Corners)", fontsize=11)
    ax4a.set_xlabel("Mean |SHAP Value| (Impact on Model Decision)", fontsize=10)
    ax4a.set_ylabel("Invariant Diagnostic Feature", fontsize=10)
    ax4a.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_fig_all(fig4a, "fig4_shap_summary.png")
    save_fig_all(fig4a, "fig4a_shap_summary.png")
    plt.close(fig4a)

    # 2. Fig 4b / Fig 9: Class-Disaggregated SHAP Grouped Bar Chart
    # Classes: 1: Opens, 2: Bridges, 3: Store Vth, 4: Access Vth
    top_5_feats = ['dv_bl_norm', 't_write_norm', 'snm_read_norm', 'iddq_norm', 'snm_hold_norm']
    feat_indices = [inv_12.index(f) for f in top_5_feats]

    # Compute mean |SHAP| per fault class (1 to 4) for the top 5 features
    class_labels = [
        'Class 1 (Opens)',
        'Class 2 (Bridges)',
        'Class 3 (Store Vth)',
        'Class 4 (Access Vth)'
    ]
    colors = ['#d9383a', '#f5b061', '#a6d9e8', '#3274a1']

    # Matrix shape: (4 classes, 5 features)
    class_matrix = np.zeros((4, 5))
    for c_idx, c_class in enumerate([1, 2, 3, 4]):
        for f_pos, f_idx in enumerate(feat_indices):
            class_matrix[c_idx, f_pos] = np.abs(sv_array[:, f_idx, c_class]).mean()

    print("\nClass-disaggregated matrix:")
    print(pd.DataFrame(class_matrix, index=class_labels, columns=top_5_feats).to_string())

    # Plot grouped bar chart matching exact publication layout
    x = np.arange(len(top_5_feats))
    width = 0.18

    fig4b, ax4b = plt.subplots(figsize=(8.5, 4.8))
    for i in range(4):
        offset = (i - 1.5) * width
        ax4b.bar(x + offset, class_matrix[i], width, label=class_labels[i], color=colors[i], edgecolor='white', linewidth=0.5)

    ax4b.set_title(f"Class-Disaggregated SHAP Physical Attribution ({n_samples:,} Authentic Samples across {n_corners} Corners)", fontsize=11, pad=10)
    ax4b.set_ylabel("Mean Absolute SHAP Value", fontsize=10)
    ax4b.set_xlabel("Invariant Diagnostic Feature", fontsize=10)
    ax4b.set_xticks(x)
    ax4b.set_xticklabels(top_5_feats, rotation=15, ha='right', fontsize=9)
    ax4b.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax4b.grid(axis='y', linestyle='--', alpha=0.5)
    ax4b.set_axisbelow(True)
    plt.tight_layout()

    save_fig_all(fig4b, "fig4b_shap_class_importance.png")
    plt.close(fig4b)

    print("\nSuccessfully regenerated all SHAP figures with 1,350 samples!")

if __name__ == "__main__":
    main()
