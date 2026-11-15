"""Read-only project audit; writes only into this review directory."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'data/sram_fault_dataset.csv')
print('Loaded dataset', df.shape, flush=True)
raw5 = ['i_ddq_uA','t_write_ps','dv_bl_strobe_mV','snm_hold_v','snm_read_v']
inv8 = ['iddq_norm','t_write_norm','dv_bl_norm','snm_hold_norm','snm_read_norm','read_write_delay_ratio','snm_ratio','dv_bl_vdd_ratio']
flags = ['hold_pass','write_pass','read_stable_pass','is_bistable']
raw12 = raw5 + ['i_write_peak_mA','t_sense_ps','v_bump_mV'] + flags
results = {'dataset_sha256': hashlib.sha256((ROOT/'data/sram_fault_dataset.csv').read_bytes()).hexdigest(),
           'shape': list(df.shape), 'classes': df.fault_class.value_counts().sort_index().to_dict(),
           'nulls': int(df.isna().sum().sum()), 'duplicate_sample_ids': int(df.sample_id.duplicated().sum()),
           'nonconverged_by_process': df[df.converged==0].groupby('process_corner').size().to_dict()}
nom = df[(df.corner_id=='TT_C_27C_1P0V') & (df.fault_class==0)]
stds = nom[raw5].std().replace(0,1e-6)
binary_pred = (abs((df[raw5]-nom[raw5].mean())/stds)>3).any(axis=1).astype(int)
results['sigma_baseline'] = {'healthy_calibration_n':len(nom), 'undefined_stds':int(stds.isna().sum()),
    'predicted_faults':int(binary_pred.sum()), 'accuracy':accuracy_score(df.fault_class>0,binary_pred),
    'always_faulty_accuracy':float((df.fault_class>0).mean())}
results['nominal_healthy_metrics'] = nom[raw5].to_dict('records')
faults = df[df.fault_class>0].copy()
faults['rank'] = faults.groupby('fault_class').severity_value.rank(method='dense').astype(int)
results['severity_rank_extremes'] = {str(k):faults[faults['rank']==k].groupby('fault_class').severity_value.unique().apply(list).to_dict() for k in [1,6]}
results['loso_healthy_train_test_overlap'] = int((df.fault_class==0).sum())
results['thermal_trend_examples'] = df[(df.fault_class==0)&(df.vdd_v==1.0)][['process_corner','temperature_c','i_ddq_uA']].to_dict('records')
tex = (ROOT/'Research_paper.tex').read_text(encoding='utf-8')
cites = set(k.strip() for s in re.findall(r'\\cite\{([^}]+)\}',tex) for k in s.split(','))
bibs = set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
results['undefined_citations'] = sorted(cites-bibs)
results['missing_root_figures'] = [x for x in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex) if not (ROOT/x).exists()]
results['simulation_inventory'] = {}
for p in ['TT','FF','SS']:
    base = ROOT/'data/sim_work'/p
    entry = {}
    for sub in ['logs','ref_log','raw','ref_raw','testbenches']:
        files = list((base/sub).glob('*'))
        entry[sub] = len([f for f in files if f.is_file()])
    for sub in ['logs','ref_log']:
        failed = []
        for file in sorted((base/sub).glob('*.log'))[:10]:
            s = file.read_text(encoding='utf-8', errors='replace').lower()
            if any(k in s for k in ['time step too small','singular matrix','simulation aborted','fatal error']):
                failed.append(file.name)
        entry[sub+'_sampled_log_count'] = min(10, len(list((base/sub).glob('*.log'))))
        entry[sub+'_sampled_fatal_marker_count'] = len(failed)
        entry[sub+'_fatal_examples'] = failed[:5]
    results['simulation_inventory'][p] = entry
archive = pd.read_csv(ROOT/'data/sram_fault_dataset_45corner_analytical_archive.csv')
results['analytical_archive'] = {'same_shape':archive.shape==df.shape,
    'equal_numeric_cells':int((archive.select_dtypes('number')==df.select_dtypes('number')).sum().sum()),
    'numeric_cells':int(df.select_dtypes('number').size)}
results['loco'] = {}
pred_records=[]
for label,cols in [('raw12',raw12),('invariant12',inv8+flags),('invariant12_converged',inv8+flags)]:
    data = df[df.converged==1] if label.endswith('_converged') else df
    all_y,all_p,accs=[],[],[]
    for corner in sorted(data.corner_id.unique()):
        train,test = data[data.corner_id!=corner],data[data.corner_id==corner]
        clf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,n_jobs=1)
        clf.fit(train[cols],train.fault_class)
        p=clf.predict(test[cols])
        all_y.extend(test.fault_class);all_p.extend(p)
        accs.append(accuracy_score(test.fault_class,p))
        if label=='invariant12':
            pred_records.extend(dict(sample_id=s,truth=int(y),prediction=int(z),converged=int(c)) for s,y,z,c in zip(test.sample_id,test.fault_class,p,test.converged))
    results['loco'][label]={'n':len(data),'accuracy':accuracy_score(all_y,all_p),
        'balanced_accuracy':balanced_accuracy_score(all_y,all_p),'macro_f1':f1_score(all_y,all_p,average='macro'),
        'corner_min':min(accs),'perfect_corners':sum(a==1 for a in accs),'confusion_matrix':confusion_matrix(all_y,all_p).tolist()}
    print(label,results['loco'][label],flush=True)
pd.DataFrame(pred_records).to_csv(OUT/'rf_loco_predictions.csv',index=False)
(OUT/'audit_results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
print(json.dumps(results,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),flush=True)
