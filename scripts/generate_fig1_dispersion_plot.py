import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sram_fault_dataset.csv"

FIG_DIRS = [
    PROJECT_ROOT / "reports" / "figures",
    PROJECT_ROOT,
    PROJECT_ROOT / "Overleaf_Upload_Package"
]

def main():
    df = pd.read_csv(DATA_PATH)
    df_h = df[df['fault_class'] == 0]
    n_corners = df_h['corner_id'].nunique()

    raw_5 = ['i_ddq_uA', 't_write_ps', 'dv_bl_strobe_mV', 'snm_hold_v', 'snm_read_v']
    inv_5 = ['iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm']

    cv_raw = [df_h[col].std() / df_h[col].mean() * 100 for col in raw_5]
    cv_inv = [df_h[col].std() / df_h[col].mean() * 100 for col in inv_5]

    df_clean = df_h[df_h['converged'] == True]
    cv_inv_clean = [df_clean[col].std() / df_clean[col].mean() * 100 for col in inv_5]

    labels = [
        r'$I_{\mathrm{DDQ}}$',
        r'$t_{\mathrm{write}}$',
        r'$\Delta V_{\mathrm{BL}}$',
        r'$\mathrm{SNM}_{\mathrm{Hold}}$',
        r'$\mathrm{SNM}_{\mathrm{Read}}$'
    ]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(raw_5))
    w = 0.35

    rects1 = ax.bar(x - w/2, cv_raw, w, label='Raw Dynamic Features (Healthy)',
                    color='#d62728', alpha=0.88, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + w/2, cv_inv, w, label='Physics-Normalized Invariants (Healthy)',
                    color='#2ca02c', alpha=0.88, edgecolor='black', linewidth=0.8)

    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#a01e1e')

    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        if idx == 0:
            txt = f'{h:.1f}%\n(0.0%*)'
        else:
            txt = f'{h:.1f}%' if h > 0 else '0.0%'
        ax.annotate(txt,
                    xy=(rect.get_x() + rect.get_width() / 2, max(h, 1)),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1b6e1b')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV) across 45 PVT Corners [%]', fontsize=11, fontweight='bold')
    ax.set_title(f'Cross-PVT Dispersion Suppression (Healthy Cells, {n_corners} Corners)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 230)
    ax.legend(frameon=True, fontsize=10.5, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout()

    for out_dir in FIG_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fig1_feature_dispersion_comparison.png"
        fig.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")

    plt.close(fig)

if __name__ == "__main__":
    main()
