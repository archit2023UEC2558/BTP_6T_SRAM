import pandas as pd
import numpy as np

df = pd.read_csv('data/sram_fault_dataset.csv')
print(f"Total rows: {len(df)}")
print(f"Total corners: {df['corner_id'].nunique()}")
print(f"Counts per corner: min={df.groupby('corner_id').size().min()}, max={df.groupby('corner_id').size().max()}")
print("\nProcess corner breakdown:")
print(df['process_corner'].value_counts())
print("\nConvergence breakdown:")
print(df.groupby('process_corner')['converged'].value_counts())

healthy = df[df['fault_class'] == 0]
print(f"\nHealthy count: {len(healthy)} (expected 45)")
print("Healthy iddq_norm summary:")
print(healthy['iddq_norm'].describe())

print("\nHealthy raw t_write_ps across TT, FF, SS:")
for pc in ['TT', 'FF', 'SS']:
    sub = healthy[healthy['process_corner'] == pc]
    print(f"  {pc}: mean={sub['t_write_ps'].mean():.1f}ps, std={sub['t_write_ps'].std():.1f}ps, min={sub['t_write_ps'].min():.1f}ps, max={sub['t_write_ps'].max():.1f}ps")

print("\nHealthy raw i_ddq_uA across TT, FF, SS:")
for pc in ['TT', 'FF', 'SS']:
    sub = healthy[healthy['process_corner'] == pc]
    print(f"  {pc}: mean={sub['i_ddq_uA'].mean():.4e}uA, std={sub['i_ddq_uA'].std():.4e}uA, min={sub['i_ddq_uA'].min():.4e}uA, max={sub['i_ddq_uA'].max():.4e}uA")

print("\nHealthy raw snm_read_v across TT, FF, SS:")
for pc in ['TT', 'FF', 'SS']:
    sub = healthy[healthy['process_corner'] == pc]
    print(f"  {pc}: mean={sub['snm_read_v'].mean():.4f}V, std={sub['snm_read_v'].std():.4f}V, min={sub['snm_read_v'].min():.4f}V, max={sub['snm_read_v'].max():.4f}V")

# Check if there are non-converged healthy runs
not_conv_healthy = healthy[healthy['converged'] == 0]
print(f"\nNon-converged healthy cells: {len(not_conv_healthy)}")
if len(not_conv_healthy) > 0:
    print(not_conv_healthy[['sample_id', 'corner_id', 'i_ddq_uA', 'iddq_norm', 'converged']])
