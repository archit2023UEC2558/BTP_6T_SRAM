import pandas as pd
import numpy as np

df = pd.read_csv('data/sram_fault_dataset.csv')
healthy = df[df['fault_class'] == 0]
print(f"Healthy cells count: {len(healthy)}")

metrics = [
    ('i_ddq_uA', 'iddq_norm', 'IDDQ (uA)'),
    ('t_write_ps', 't_write_norm', 't_write (ps)'),
    ('dv_bl_strobe_mV', 'dv_bl_norm', 'dv_bl (mV)'),
    ('snm_hold_v', 'snm_hold_norm', 'SNM_Hold (V)'),
    ('snm_read_v', 'snm_read_norm', 'SNM_Read (V)')
]

for raw_col, norm_col, label in metrics:
    raw_m = healthy[raw_col].mean()
    raw_s = healthy[raw_col].std()
    raw_cv = (raw_s / raw_m) * 100 if raw_m != 0 else 0

    norm_m = healthy[norm_col].mean()
    norm_s = healthy[norm_col].std()
    norm_cv = (norm_s / norm_m) * 100 if norm_m != 0 else 0

    red = ((raw_cv - norm_cv) / raw_cv) * 100 if raw_cv != 0 else 0

    print(f"{label:16s} | Raw: {raw_m:9.4f} +/- {raw_s:8.4f} (CV {raw_cv:6.2f}%) | "
          f"Norm: {norm_m:7.4f} +/- {norm_s:7.4f} (CV {norm_cv:6.2f}%) | Reduction: {red:6.2f}%")

# Clean subset for IDDQ excluding the 1 sentinel run
clean_healthy = healthy[healthy['converged'] == 1]
print(f"\nClean healthy cells count: {len(clean_healthy)}")
raw_m = clean_healthy['i_ddq_uA'].mean()
raw_s = clean_healthy['i_ddq_uA'].std()
raw_cv = (raw_s / raw_m) * 100
norm_m = clean_healthy['iddq_norm'].mean()
norm_s = clean_healthy['iddq_norm'].std()
norm_cv = (norm_s / norm_m) * 100
red = ((raw_cv - norm_cv) / raw_cv) * 100
print(f"IDDQ (Clean {len(clean_healthy)}/45) | Raw: {raw_m:9.4f} +/- {raw_s:8.4f} (CV {raw_cv:6.2f}%) | "
      f"Norm: {norm_m:7.4f} +/- {norm_s:7.4f} (CV {norm_cv:6.2f}%) | Reduction: {red:6.2f}%")
