"""Rolling (causal) alarm thresholds vs the fixed validation threshold, and ETF back-tests.

Usage (from the repository root):
    python -m reversal_v2.rolling <run_root> --out <csv_prefix> [--models GRU-hazard,MLP-hazard,logreg,ENS-hazard]
        [--etf]            use SPY/QQQ/DIA/IWM opens instead of index opens for the back-test

Threshold rules for the 10-day reversal alarm (all causal):
  fixed  : quantile of the validation outputs at (1 - validation base rate)   (what 13-18 used)
  roll250: same quantile, but of the model's own outputs over the trailing 250 days
           (validation outputs serve as warm-up), recomputed every day
Exposure rule `graded` as in 13 §4.1, with `top` = trailing max instead of the validation max.
"""
import argparse
import os

import numpy as np
import pandas as pd

from reversal_v2 import data as D
from reversal_v2.analyze import COST, load_run, perf, use_run_label

ETF = {'GSPC': 'SPY', 'IXIC': 'QQQ', 'DJI': 'DIA', 'RUT': 'IWM'}
RNG = np.random.default_rng(0)


def thresholds(val_e, test_e, val_rev, rule, window=250):
    q = 1 - (val_rev <= 10).mean()
    if rule == 'fixed':
        thr = np.full(len(test_e), np.quantile(val_e, q))
        top = np.full(len(test_e), val_e.max())
        return thr, top
    s = pd.Series(np.concatenate([val_e[-window:], test_e]))
    thr = s.rolling(window, min_periods=50).quantile(q).values[-len(test_e):]
    top = s.rolling(window, min_periods=50).max().values[-len(test_e):]
    return thr, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='GRU-hazard,MLP-hazard,logreg,ENS-hazard')
    ap.add_argument('--symbols', default='GSPC,IXIC,DJI,RUT')
    ap.add_argument('--etf', action='store_true')
    args = ap.parse_args()
    use_run_label(args.run_root)
    rows, bt = [], []
    for sym in args.symbols.split(','):
        folds = load_run(args.run_root if sym == 'GSPC' else os.path.join(args.run_root, sym))
        for f in folds:
            if all(m in f['preds'] for m in ('GRU-hazard', 'MLP-hazard')):
                f['preds']['ENS-hazard'] = {k: np.mean([f['preds'][m][k] for m in ('GRU-hazard', 'MLP-hazard')], 0)
                                            for k in f['preds']['GRU-hazard']}
        prices = None
        if args.etf:
            prices = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'research', 'data', 'raw_v3', ETF[sym] + '.csv'),
                                 index_col=0, parse_dates=True)['Open']
        for m in args.models.split(','):
            for rule in ['fixed', 'roll250']:
                stats = dict(tp=0, fp=0, caught=0, total=0, days=0, alarm_rates=[])
                r_strat, r_bh = [], []
                for f in folds:
                    lab = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('test_')}
                    val = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('val_')}
                    p = f['preds'][m]
                    e, rev = p['test_event10'], lab['rev']
                    thr, top = thresholds(p['val_event10'], e, val['rev'], rule)
                    alarm = e >= thr
                    ev = rev <= 10
                    stats['tp'] += (alarm & ev).sum(); stats['fp'] += (alarm & ~ev).sum(); stats['days'] += len(e)
                    stats['alarm_rates'].append(alarm.mean())
                    for t in np.where(rev == 1)[0]:
                        stats['total'] += 1
                        stats['caught'] += alarm[max(0, t - 9):t + 1].any()
                    # graded exposure
                    up = p['test_trend'][:, 0] < 0.5
                    expo = np.where(up, 1 - np.clip((e - thr) / np.maximum(top - thr, 1e-9), 0, 1), 1.0)
                    dates = pd.to_datetime(lab['dates'])
                    if prices is not None:
                        op = prices.reindex(dates).values
                        if np.isnan(op).mean() > 0.02:
                            continue
                        op = pd.Series(op).ffill().values
                    else:
                        op = lab['open']
                    ret = op[2:] / op[1:-1] - 1
                    pos = expo[:-2]
                    turn = np.abs(np.diff(np.concatenate([[1.0], pos])))
                    r_strat.append(pos * ret - turn * COST); r_bh.append(ret)
                ar = np.array(stats['alarm_rates'])
                rows.append(dict(symbol=sym, model=m, rule=rule, precision=stats['tp'] / max(stats['tp'] + stats['fp'], 1),
                                 recall_events=stats['caught'] / max(stats['total'], 1),
                                 false_alarm_days_per_year=stats['fp'] / stats['days'] * 252,
                                 alarm_rate_mean=ar.mean(), alarm_rate_std=ar.std(), alarm_rate_max=ar.max()))
                if r_strat:
                    rs, rb = np.concatenate(r_strat), np.concatenate(r_bh)
                    ps, pb = perf(rs), perf(rb)
                    # random benchmark: same exposure path, 20-day blocks shuffled
                    bt.append(dict(symbol=sym, model=m, rule=rule, instrument=ETF[sym] if args.etf else sym,
                                   sharpe=ps['sharpe'], sharpe_bh=pb['sharpe'], cagr=ps['cagr'], cagr_bh=pb['cagr'],
                                   max_dd=ps['max_dd'], max_dd_bh=pb['max_dd'],
                                   years=len(rs) / 252))
    alarms, backtest = pd.DataFrame(rows), pd.DataFrame(bt)
    alarms.to_csv(args.out + '_alarms.csv', index=False)
    backtest.to_csv(args.out + '_backtest.csv', index=False)
    pd.set_option('display.width', 250)
    print(alarms.round(3).to_string())
    print(backtest.round(3).to_string())


if __name__ == '__main__':
    main()
