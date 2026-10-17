import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/sram_fault_dataset.csv')

inv_cols = [
    'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
    'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio',
    'hold_pass', 'write_pass', 'read_stable_pass', 'is_bistable'
]

# Extract severity tag (S0 to S6)
def get_sev(sid):
    parts = sid.split('_')
    for p in parts:
        if p.startswith('S') and len(p) == 2 and p[1].isdigit():
            return p
    return 'S0'

df['severity_tag'] = df['sample_id'].apply(get_sev)
print('Severity tag breakdown:\n', df['severity_tag'].value_counts())

# LOSO Test 1: Train on mild-moderate (S1-S4), test on extreme severities (S5-S6)
train_df = df[df['severity_tag'].isin(['S0', 'S1', 'S2', 'S3', 'S4'])]
test_df = df[df['severity_tag'].isin(['S5', 'S6'])]

clf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf1.fit(train_df[inv_cols], train_df['fault_class'])
preds1 = clf1.predict(test_df[inv_cols])
acc1 = accuracy_score(test_df['fault_class'], preds1)
print(f'LOSO Extrapolation (Train S0-S4, Test unseen S5-S6): {acc1*100:.2f}% ({np.sum(preds1 == test_df["fault_class"])}/{len(test_df)})')

# LOSO Test 2: Interpolation - Train on boundaries (S0, S1, S2, S5, S6), test on intermediate (S3, S4)
train_df2 = df[df['severity_tag'].isin(['S0', 'S1', 'S2', 'S5', 'S6'])]
test_df2 = df[df['severity_tag'].isin(['S3', 'S4'])]

clf2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf2.fit(train_df2[inv_cols], train_df2['fault_class'])
preds2 = clf2.predict(test_df2[inv_cols])
acc2 = accuracy_score(test_df2['fault_class'], preds2)
print(f'LOSO Interpolation (Train S0-S2+S5-S6, Test unseen S3-S4): {acc2*100:.2f}% ({np.sum(preds2 == test_df2["fault_class"])}/{len(test_df2)})')
