import json, subprocess, csv, sys
import os
repo=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
files=subprocess.run(['git','-C',repo,'ls-tree','-r','--name-only','dd800b2'],capture_output=True,text=True).stdout.split()
rows=[]
for f in files:
    if not f.endswith('summary.json'): continue
    try: d=json.loads(subprocess.run(['git','-C',repo,'show','dd800b2:'+f],capture_output=True,text=True).stdout)
    except Exception as e: rows.append({'path':f,'err':str(e)}); continue
    p=d.get('usingData',{})
    def g(k):
        v=d.get(k)
        if isinstance(v,str):
            try: v=json.loads(v)
            except: pass
        return v
    t=g('trend_confusion_matrix_info') or {}
    s=g('signal_confusion_matrix_info') or {}
    r=g('reversed_trend_confusion_matrix_info') or {}
    rows.append(dict(path=f.rsplit('/reports',1)[0], model=p.get('model_type'), trend_in_feat='Trend' in p.get('feature_cols',[]),
      nfeat=len(p.get('feature_cols',[])), lb=p.get('look_back'), ps=p.get('predict_steps'), lr=p.get('learning_rate'),
      w_after=p.get('weight_after_reversal'), split=f"{p.get('train_split_ratio')}/{p.get('val_split_ratio')}", period=f"{p.get('start_date')}~{p.get('stop_date')}",
      trend_acc=(t.get('Accuracy') or {}).get('0'), trend_f1=(t.get('F1 Score') or {}).get('0'),
      rev_acc=(r.get('Accuracy') or {}).get('0'), rev_f1=(r.get('F1 Score') or {}).get('0'),
      sig_f1=(s.get('F1 Score') or {}).get('0'), roc_auc=g('roc_auc'), days_diff=g('pred_days_difference_abs_mean'), in_adv=g('pred_in_advance'),
      bt=str(g('backtesting_report'))[:200]))
w=csv.DictWriter(sys.stdout, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
w.writeheader(); w.writerows(rows)
