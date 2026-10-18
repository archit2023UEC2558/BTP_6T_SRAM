import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

df = pd.read_csv('data/sram_fault_dataset.csv')

def get_sev(sid):
    parts = sid.split('_')
    for p in parts:
        if p.startswith('S') and len(p) == 2 and p[1].isdigit():
            return p
    return 'S0'

df['severity_tag'] = df['sample_id'].apply(get_sev)

feature_sets = {
    'Full Invariant (12 Features)': [
        'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
        'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio',
        'hold_pass', 'write_pass', 'read_stable_pass', 'is_bistable'
    ],
    'Parametric Ratios Only (8 Features, No Flags)': [
        'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
        'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio'
    ],
    'Matched Analog Ratios Only (5 Features)': [
        'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm'
    ],
    'Functional Flags Only (4 Features)': [
        'hold_pass', 'write_pass', 'read_stable_pass', 'is_bistable'
    ]
}

corners = df['corner_id'].unique()
results = []

for name, cols in feature_sets.items():
    all_preds = []
    all_trues = []
    corner_accs = []
    
    for c in corners:
        train_mask = (df['corner_id'] != c) & (df['severity_tag'].isin(['S0', 'S1', 'S2', 'S5', 'S6']))
        test_mask = (df['corner_id'] == c) & (df['severity_tag'].isin(['S3', 'S4']))
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(train_df[cols], train_df['fault_class'])
        preds = clf.predict(test_df[cols])
        
        all_preds.extend(preds)
        all_trues.extend(test_df['fault_class'])
        corner_accs.append(accuracy_score(test_df['fault_class'], preds))
        
    acc = accuracy_score(all_trues, all_preds)
    bal_acc = balanced_accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average='macro')
    mean_corner = np.mean(corner_accs)
    min_corner = np.min(corner_accs)
    max_corner = np.max(corner_accs)
    
    print(f"\n--- {name} ---")
    print(f"Overall Accuracy: {acc*100:.2f}% ({np.sum(np.array(all_preds) == np.array(all_trues))}/{len(all_trues)})")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Macro-F1: {f1*100:.2f}%")
    print(f"Mean Corner: {mean_corner*100:.2f}% (Floor: {min_corner*100:.2f}%, Max: {max_corner*100:.2f}%)")
    
    results.append({
        'Feature_Set': name,
        'Accuracy': acc,
        'Balanced_Acc': bal_acc,
        'Macro_F1': f1,
        'Mean_Corner_Acc': mean_corner,
        'Worst_Corner_Floor': min_corner,
        'Best_Corner': max_corner
    })

res_df = pd.DataFrame(results)
res_df.to_csv('reports/final_results/joint_holdout_experiment_c.csv', index=False)
print("\nSaved joint holdout results to reports/final_results/joint_holdout_experiment_c.csv")
