import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/sram_fault_dataset.csv')
print(f"Loaded {len(df)} samples across {df['corner_id'].nunique()} corners.")

corners = df['corner_id'].unique()
healthy = df[df['fault_class'] == 0].set_index('corner_id')

raw_cols = ['i_ddq_uA', 't_write_ps', 'dv_bl_strobe_mV', 'snm_hold_v', 'snm_read_v']

# Mismatch standard deviations to test: 0% to 10%
mismatch_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
results = []

np.random.seed(42)

for sigma in mismatch_levels:
    # Build mismatched ratios: X_norm = X_meas / (X_ref * (1 + N(0, sigma)))
    noisy_features = []
    for _, row in df.iterrows():
        c = row['corner_id']
        ref = healthy.loc[c]
        row_dict = {}
        for col in raw_cols:
            denom = ref[col] if abs(ref[col]) > 1e-12 else 1e-12
            # Add independent Gaussian perturbation to reference tracking
            noise_factor = 1.0 + np.random.normal(0, sigma) if sigma > 0 else 1.0
            noisy_ref = denom * max(noise_factor, 0.1)
            row_dict[col + '_norm'] = row[col] / noisy_ref
        noisy_features.append(row_dict)
    
    noisy_df = pd.concat([df[['corner_id', 'fault_class']], pd.DataFrame(noisy_features)], axis=1)
    norm_cols = [col + '_norm' for col in raw_cols]
    
    corner_accs = []
    for c in corners:
        train_df = noisy_df[noisy_df['corner_id'] != c]
        test_df = noisy_df[noisy_df['corner_id'] == c]
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        clf.fit(train_df[norm_cols], train_df['fault_class'])
        preds = clf.predict(test_df[norm_cols])
        corner_accs.append(accuracy_score(test_df['fault_class'], preds))
        
    mean_acc = np.mean(corner_accs)
    min_acc = np.min(corner_accs)
    print(f"Mismatch sigma={sigma*100:4.1f}%: LOCO Mean Accuracy = {mean_acc*100:.2f}%, Worst-Corner Floor = {min_acc*100:.2f}%")
    results.append({'sigma_percent': sigma * 100, 'mean_accuracy': mean_acc * 100, 'worst_corner': min_acc * 100})

res_df = pd.DataFrame(results)
res_df.to_csv('reports/final_results/reference_mismatch_sensitivity.csv', index=False)
print("Saved sensitivity results to reports/final_results/reference_mismatch_sensitivity.csv")

# Plot figure
plt.figure(figsize=(7, 4.5))
plt.plot(res_df['sigma_percent'], res_df['mean_accuracy'], 'b-o', linewidth=2, markersize=7, label='LOCO Mean Accuracy')
plt.plot(res_df['sigma_percent'], res_df['worst_corner'], 'r--s', linewidth=2, markersize=6, label='Worst-Corner Accuracy Floor')
plt.axhline(87.26, color='gray', linestyle=':', label='Raw Features Baseline (87.26%)')
plt.title('Diagnostic Robustness Under Reference-Cell Device Mismatch', fontsize=12, pad=10)
plt.xlabel('Local Intra-Die Reference Mismatch $\\sigma$ (%)', fontsize=11)
plt.ylabel('Classification Accuracy (%)', fontsize=11)
plt.ylim(70, 102)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', framealpha=0.9)
plt.tight_layout()
plt.savefig('fig6_reference_mismatch_sensitivity.png', dpi=300)
plt.savefig('reports/figures/fig6_reference_mismatch_sensitivity.png', dpi=300)
print("Saved fig6_reference_mismatch_sensitivity.png")
