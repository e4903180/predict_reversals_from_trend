"""Usage: python compare_runs.py <run_dir> [<run_dir> ...]  (each containing outputs/reports/*.json)"""
import json, io, sys, pandas as pd, numpy as np
def load(run, name):
    return json.load(open(f'{run}/outputs/reports/{name}.json'))
def J(v):
    return json.loads(v) if isinstance(v, str) else v
def summarize(d):
    t = J(d['trend_confusion_matrix_info']); o3 = J(d['overall_reversals_confusion_matrix_info']); c3 = J(d['class_reversals_confusion_matrix_info'])
    rd = pd.read_json(io.StringIO(d['reverse_difference']))
    td = pd.read_json(io.StringIO(d['trade_details'])); lo = pd.read_json(io.StringIO(d['trade_details_long_only']))
    yp = np.array(d['y_preds']); yt = np.array(d['y_data'])
    return {
      'roc_auc': round(float(d['roc_auc']),4), 'pr_auc': round(float(d['pr_auc']),4),
      'trend_acc': round(t['Accuracy']['0'],4), 'trend_f1(up)': round(t['F1 Score']['0'],4),
      'pred_up_share': round(float((yp<=0.5).mean()),3), 'true_up_share': round(float((yt==0).mean()),3),
      'rev3_macroF1': round(o3['F1 Score']['Macro'],4), 'peak_F1': round(c3['F1 Score']['Peak'],4), 'valley_F1': round(c3['F1 Score']['Valley'],4),
      'true_rev_windows': len(rd), 'pred_type_correct': int(rd['reverse_signal_correct'].sum()), 'in_range±5': int(d['reverse_in_range_num']),
      'n_trades': len(td), 'longshort_profit': round(td['Profit'].iloc[-1],1) if len(td) else 0, 'win': int((td.Outcome=='win').sum()),
      'longonly_profit': round(lo['Profit'].iloc[-1],1) if len(lo) else 0, 'first_trade': str(td.index.min().date()) if len(td) else None, 'last_trade': str(td.index.max().date()) if len(td) else None}
rows = {}
for run, label in [(r, r) for r in sys.argv[1:]]:
    for name in ['val_summary','summary']:
        rows[(label, name)] = summarize(load(run, name))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string())
