import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

df = pd.read_csv('data/sram_fault_dataset.csv')
print(f"Loaded dataset: {len(df)} rows across {df['corner_id'].nunique()} corners.")

corners = df['corner_id'].unique()

# -------------------------------------------------------------
# 1. PARAMETRIC-ONLY FEATURE ABLATION (Removing all 4 status flags)
# -------------------------------------------------------------
full_inv_cols = [
    'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
    'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio',
    'hold_pass', 'write_pass', 'read_stable_pass', 'is_bistable'
]

parametric_only_cols = [
    'iddq_norm', 't_write_norm', 'dv_bl_norm', 'snm_hold_norm', 'snm_read_norm',
    'read_write_delay_ratio', 'snm_ratio', 'dv_bl_vdd_ratio'
]

def run_loco(feature_cols, data_df):
    all_preds = []
    all_trues = []
    corner_accs = []
    
    for c in corners:
        train_df = data_df[data_df['corner_id'] != c]
        test_df = data_df[data_df['corner_id'] == c]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(train_df[feature_cols], train_df['fault_class'])
        preds = clf.predict(test_df[feature_cols])
        
        all_preds.extend(preds)
        all_trues.extend(test_df['fault_class'])
        corner_accs.append(accuracy_score(test_df['fault_class'], preds))
        
    acc = accuracy_score(all_trues, all_preds)
    bal_acc = balanced_accuracy_score(all_trues, all_preds)
    macro_f1 = f1_score(all_trues, all_preds, average='macro')
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'corner_mean': np.mean(corner_accs),
        'corner_std': np.std(corner_accs),
        'corner_min': np.min(corner_accs),
        'corner_max': np.max(corner_accs),
        'trues': all_trues,
        'preds': all_preds
    }

print("\n--- Running LOCO: Full Invariant (12 features) ---")
res_full = run_loco(full_inv_cols, df)
print(f"Full: Acc={res_full['accuracy']*100:.2f}%, BalAcc={res_full['balanced_accuracy']*100:.2f}%, Macro-F1={res_full['macro_f1']*100:.2f}%")

print("\n--- Running LOCO: Purely Parametric Only (8 features, NO status flags) ---")
res_param = run_loco(parametric_only_cols, df)
print(f"Param-Only: Acc={res_param['accuracy']*100:.2f}%, BalAcc={res_param['balanced_accuracy']*100:.2f}%, Macro-F1={res_param['macro_f1']*100:.2f}%")
print(f"Corner Mean={res_param['corner_mean']*100:.2f}% (Min={res_param['corner_min']*100:.2f}%, Max={res_param['corner_max']*100:.2f}%)")

# -------------------------------------------------------------
# 2. STRICTLY MATCHED 5-FEATURE NORMALIZATION COMPARISON (Raw vs Diff vs Log-Ratio vs Ratio)
# -------------------------------------------------------------
raw_5 = ['i_ddq_uA', 't_write_ps', 'dv_bl_strobe_mV', 'snm_hold_v', 'snm_read_v']
healthy = df[df['fault_class'] == 0].set_index('corner_id')

diff_rows = []
log_ratio_rows = []
ratio_rows = []

for _, row in df.iterrows():
    c = row['corner_id']
    ref = healthy.loc[c]
    d_dict, lr_dict, r_dict = {}, {}, {}
    for col in raw_5:
        # Difference
        d_dict[col + '_diff'] = row[col] - ref[col]
        # Ratio
        denom = ref[col] if abs(ref[col]) > 1e-12 else 1e-12
        r_val = max(row[col] / denom, 1e-12)
        r_dict[col + '_ratio'] = r_val
        # Log-Ratio
        lr_dict[col + '_logratio'] = np.log(r_val)
    diff_rows.append(d_dict)
    log_ratio_rows.append(lr_dict)
    ratio_rows.append(r_dict)

df_ablation = pd.concat([
    df[['corner_id', 'fault_class', 'converged']],
    df[raw_5],
    pd.DataFrame(diff_rows),
    pd.DataFrame(log_ratio_rows),
    pd.DataFrame(ratio_rows)
], axis=1)

diff_5 = [col + '_diff' for col in raw_5]
log_ratio_5 = [col + '_logratio' for col in raw_5]
ratio_5 = [col + '_ratio' for col in raw_5]

print("\n--- Matched 5-Feature Ablation ---")
res_raw5 = run_loco(raw_5, df_ablation)
print(f"Matched Raw (5): Acc={res_raw5['accuracy']*100:.2f}%, Min={res_raw5['corner_min']*100:.2f}%, Max={res_raw5['corner_max']*100:.2f}%")

res_diff5 = run_loco(diff_5, df_ablation)
print(f"Matched Diff (5): Acc={res_diff5['accuracy']*100:.2f}%, Min={res_diff5['corner_min']*100:.2f}%, Max={res_diff5['corner_max']*100:.2f}%")

res_lr5 = run_loco(log_ratio_5, df_ablation)
print(f"Matched Log-Ratio (5): Acc={res_lr5['accuracy']*100:.2f}%, Min={res_lr5['corner_min']*100:.2f}%, Max={res_lr5['corner_max']*100:.2f}%")

res_ratio5 = run_loco(ratio_5, df_ablation)
print(f"Matched Ratio (5): Acc={res_ratio5['accuracy']*100:.2f}%, Min={res_ratio5['corner_min']*100:.2f}%, Max={res_ratio5['corner_max']*100:.2f}%")

# -------------------------------------------------------------
# 3. CLEAN SUBSET EVALUATION (1,284 converged samples with NO timeouts)
# -------------------------------------------------------------
df_clean = df[df['converged'] == 1].copy()
res_clean = run_loco(parametric_only_cols, df_clean)
print(f"\n--- Clean Subset (1284 rows, Parametric-Only): Acc={res_clean['accuracy']*100:.2f}%, BalAcc={res_clean['balanced_accuracy']*100:.2f}%, Macro-F1={res_clean['macro_f1']*100:.2f}% ---")

# -------------------------------------------------------------
# 4. PER-CLASS METRICS AND CONFUSION MATRIX
# -------------------------------------------------------------
class_names = ['Class 0 (Healthy)', 'Class 1 (Resistive Open)', 'Class 2 (Bridging)', 'Class 3 (Storage Vth Drift)', 'Class 4 (Access Vth Drift)']

print("\n--- Detailed Per-Class Classification Report (Purely Parametric Only) ---")
print(classification_report(res_param['trues'], res_param['preds'], target_names=class_names, digits=4))

# Generate and save Confusion Matrix plot
cm = confusion_matrix(res_param['trues'], res_param['preds'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6.5))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Healthy', 'Open', 'Bridge', 'Store Vth', 'Access Vth'],
            yticklabels=['Healthy', 'Open', 'Bridge', 'Store Vth', 'Access Vth'],
            cbar_kws={'label': 'Normalized Recall'})
plt.title('Pooled Cross-PVT LOCO Confusion Matrix\n(Purely Parametric Features, No Status Flags, 1350 Samples)', fontsize=12, pad=12)
plt.xlabel('Predicted Fault Class', fontsize=11)
plt.ylabel('Ground Truth Physical Class', fontsize=11)
plt.tight_layout()

os.makedirs('reports/figures', exist_ok=True)
cm_fig_path = 'reports/figures/fig5_confusion_matrix.png'
plt.savefig(cm_fig_path, dpi=300)
plt.savefig('fig5_confusion_matrix.png', dpi=300)
print(f"Saved confusion matrix figure to {cm_fig_path} and root fig5_confusion_matrix.png")

# Save summary CSV
summary_df = pd.DataFrame([
    {'Experiment': 'Matched Raw (5 Features)', 'Accuracy': res_raw5['accuracy'], 'Balanced_Acc': res_raw5['balanced_accuracy'], 'Macro_F1': res_raw5['macro_f1'], 'Worst_Corner': res_raw5['corner_min']},
    {'Experiment': 'Matched Difference (5 Features)', 'Accuracy': res_diff5['accuracy'], 'Balanced_Acc': res_diff5['balanced_accuracy'], 'Macro_F1': res_diff5['macro_f1'], 'Worst_Corner': res_diff5['corner_min']},
    {'Experiment': 'Matched Log-Ratio (5 Features)', 'Accuracy': res_lr5['accuracy'], 'Balanced_Acc': res_lr5['balanced_accuracy'], 'Macro_F1': res_lr5['macro_f1'], 'Worst_Corner': res_lr5['corner_min']},
    {'Experiment': 'Matched Ratio (5 Features)', 'Accuracy': res_ratio5['accuracy'], 'Balanced_Acc': res_ratio5['balanced_accuracy'], 'Macro_F1': res_ratio5['macro_f1'], 'Worst_Corner': res_ratio5['corner_min']},
    {'Experiment': 'Full Invariant (12 Features)', 'Accuracy': res_full['accuracy'], 'Balanced_Acc': res_full['balanced_accuracy'], 'Macro_F1': res_full['macro_f1'], 'Worst_Corner': res_full['corner_min']},
    {'Experiment': 'Parametric-Only (8 Features, No Flags)', 'Accuracy': res_param['accuracy'], 'Balanced_Acc': res_param['balanced_accuracy'], 'Macro_F1': res_param['macro_f1'], 'Worst_Corner': res_param['corner_min']},
    {'Experiment': 'Clean Subset (Parametric-Only)', 'Accuracy': res_clean['accuracy'], 'Balanced_Acc': res_clean['balanced_accuracy'], 'Macro_F1': res_clean['macro_f1'], 'Worst_Corner': res_clean['corner_min']},
])
summary_df.to_csv('reports/final_results/controlled_ablation_summary.csv', index=False)
print("Saved controlled ablation summary to reports/final_results/controlled_ablation_summary.csv")
