"""Usage: python baselines.py <original_run_dir> <fixed_run_dir> <GSPC.csv>"""
import sys, json, io, pandas as pd, numpy as np
orig, fixed, csv = sys.argv[1:4]
from sklearn.metrics import roc_auc_score
px = pd.read_csv(csv, index_col=0, parse_dates=True)['Close']
for name in ['val_summary', 'summary']:
    o = json.load(open(f'{orig}/outputs/reports/{name}.json')); f = json.load(open(f'{fixed}/outputs/reports/{name}.json'))
    yt = np.array(f['y_data']); yp = np.array(f['y_preds']); yo = np.array(o['y_preds'])
    per = [roc_auc_score(yt[:, k], yp[:, k]) for k in range(16)]
    print(name, 'per-step AUC (fixed probs):', ' '.join(f'{a:.3f}' for a in per))
    # what the original did: every step scored with the step-1 logit
    smear = np.repeat(yp[:, 1:2], 16, axis=1); smear[:, 0] = yp[:, 0]
    print(name, 'AUC flat fixed=%.4f  | step-1 value reused for all steps=%.4f | original saved=%.4f' % (
        roc_auc_score(yt.ravel(), yp.ravel()), roc_auc_score(yt.ravel(), smear.ravel()), roc_auc_score(yt.ravel(), yo.ravel())))
    td = pd.read_json(io.StringIO(o['trade_details']))
    print(name, 'always-uptrend acc=%.4f AUC=0.5' % (yt == 0).mean())
    # period from first to last oracle trade
    s, e = td.index.min(), td.index.max()
    p = px.loc[s:e]
    print(name, f'Buy&Hold 1 share {s.date()}..{e.date()}: {p.iloc[-1]-p.iloc[0]:+.1f}  | oracle long-short: {td.Profit.iloc[-1]:+.1f}')
    # MA20/50 cross, long-short 1 share, signals at close, same period
    full = px.loc[:e]; ma20 = full.rolling(20).mean(); ma50 = full.rolling(50).mean()
    pos = np.sign(ma20 - ma50).loc[s:e]
    pnl = (pos.shift(1) * p.diff()).sum()
    print(name, f'MA20/50 cross long-short 1 share: {pnl:+.1f}, trades={int((pos.diff()!=0).sum())}')
