"""Pooled evaluation over several test indices: more reversal events, calibration, alarm rules.

Usage (from the repository root):
    python -m reversal_v2.pooled <run_root> --symbols GSPC,IXIC,DJI,RUT --out <dir> \
        [--ensemble ENS-hazard=GRU-hazard+TCN-hazard+MLP-hazard]

<run_root> is a walk-forward output written with --eval-indices: GSPC results live in
<run_root>/fold*, every other index in <run_root>/<SYM>/fold*. Thresholds and calibrators
are always fitted on the (GSPC) validation block of the same fold.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from reversal_v2.analyze import COST, age_hazard_baseline, load_run, perf, reversal_positions, use_run_label
from reversal_v2.walkforward import EVENT_M

RNG = np.random.default_rng(0)


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def persistent(alarm, k):
    """Alarm only after k consecutive raw alarms."""
    if k == 1:
        return alarm
    out = alarm.copy()
    for j in range(1, k):
        out[j:] &= alarm[:-j]
        out[:j] = False
    return out


def collect(run_root, symbols, ensembles):
    """rows[(sym, fold)] = dict(labels=test labels, val=val labels, preds={model: {...}})"""
    cells = {}
    for sym in symbols:
        folds = load_run(run_root if sym == 'GSPC' else os.path.join(run_root, sym))
        for f in folds:
            for name, members in ensembles.items():
                if all(m in f['preds'] for m in members):
                    f['preds'][name] = {k: np.mean([f['preds'][m][k] for m in members], 0) for k in f['preds'][members[0]]}
            lab = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('test_')}
            val = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('val_')}
            base = age_hazard_baseline(f['k'], lab, sym)
            f['preds']['age_hazard'] = {f'test_event{m}': base[m] for m in EVENT_M}
            cells[(sym, f['k'])] = dict(lab=lab, val=val, preds=f['preds'])
    return cells


def stratified_block_bootstrap(groups, stat, n=1000, block=20):
    """groups: list of dicts of aligned arrays; resample blocks within each group, then pool."""
    out = []
    for _ in range(n):
        parts = []
        for g in groups:
            t = len(next(iter(g.values())))
            starts = RNG.integers(0, t - block + 1, t // block + 1)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:t]
            parts.append({k: v[idx] for k, v in g.items()})
        pooled = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        v = stat(pooled)
        if v is not None:
            out.append(v)
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), (out <= 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_root')
    ap.add_argument('--symbols', default='GSPC,IXIC,DJI,RUT')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ensemble', action='append', default=[])
    ap.add_argument('--n', type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    symbols = args.symbols.split(',')
    ensembles = {s.split('=')[0]: s.split('=')[1].split('+') for s in args.ensemble}
    use_run_label(args.run_root)
    cells = collect(args.run_root, symbols, ensembles)
    models = sorted(set.intersection(*[set(c['preds']) for c in cells.values()]) - {'age_hazard'})

    # ---------------- event metrics per cell, win counts ----------------
    rows = []
    for (sym, k), c in cells.items():
        rev = c['lab']['rev']
        for m in models + ['age_hazard']:
            p = c['preds'][m]
            row = dict(symbol=sym, fold=k, model=m, n_reversals=int((rev == 1).sum()))
            for e in EVENT_M:
                y = (rev <= e).astype(int)
                row[f'event{e}_auc'] = roc_auc_score(y, p[f'test_event{e}'])
                row[f'event{e}_ap'] = average_precision_score(y, p[f'test_event{e}'])
                row[f'event{e}_base'] = y.mean()
            rows.append(row)
    cell_df = pd.DataFrame(rows)
    cell_df.to_csv(os.path.join(args.out, 'cells.csv'), index=False)
    base = cell_df[cell_df.model == 'age_hazard'].set_index(['symbol', 'fold'])
    summ = []
    for m in models + ['age_hazard']:
        d = cell_df[cell_df.model == m].set_index(['symbol', 'fold'])
        r = dict(model=m)
        for e in EVENT_M:
            r[f'event{e}_auc'] = d[f'event{e}_auc'].mean()
            r[f'event{e}_ap'] = d[f'event{e}_ap'].mean()
            wins = int((d[f'event{e}_auc'] > base[f'event{e}_auc']).sum())
            r[f'event{e}_wins'] = f'{wins}/{len(d)}'
            r[f'event{e}_sign_p'] = binomtest(wins, len(d), 0.5, alternative='greater').pvalue if m != 'age_hazard' else np.nan
        for sym in symbols:
            r[f'event10_auc_{sym}'] = d.loc[sym, 'event10_auc'].mean()
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out, 'summary.csv'), index=False)

    # ---------------- pooled stratified bootstrap ----------------
    tests = []
    for m in models:
        for e in EVENT_M:
            for ref in ['age_hazard', 'logreg']:
                if ref == m or ref not in cells[next(iter(cells))]['preds']:
                    continue
                groups = []
                for sym in symbols:
                    ks = sorted(k for s, k in cells if s == sym)
                    groups.append(dict(
                        y=np.concatenate([(cells[(sym, k)]['lab']['rev'] <= e).astype(int) for k in ks]),
                        a=np.concatenate([cells[(sym, k)]['preds'][m][f'test_event{e}'] for k in ks]),
                        b=np.concatenate([cells[(sym, k)]['preds'][ref][f'test_event{e}'] for k in ks])))
                stat = lambda g: roc_auc_score(g['y'], g['a']) - roc_auc_score(g['y'], g['b']) if 0 < g['y'].mean() < 1 else None
                mean, lo, hi, p = stratified_block_bootstrap(groups, stat, n=args.n)
                tests.append(dict(model=m, event=e, versus=ref, auc_diff=mean, ci_low=lo, ci_high=hi, p_le_0=p))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(args.out, 'bootstrap.csv'), index=False)

    # ---------------- calibration (isotonic on the validation block) ----------------
    cal_rows = []
    for m in models:
        ys, raw, cal = [], [], []
        for (sym, k), c in cells.items():
            p = c['preds'][m]
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            iso.fit(p['val_event10'], (c['val']['rev'] <= 10).astype(int))
            ys.append((c['lab']['rev'] <= 10).astype(int)); raw.append(p['test_event10']); cal.append(iso.predict(p['test_event10']))
        y, raw, cal = map(np.concatenate, (ys, raw, cal))
        cal_rows.append(dict(model=m, brier_raw=brier_score_loss(y, raw), brier_cal=brier_score_loss(y, cal),
                             brier_climatology=brier_score_loss(y, np.full_like(raw, y.mean())),
                             ece_raw=ece(y, raw), ece_cal=ece(y, cal), mean_pred_raw=raw.mean(), mean_pred_cal=cal.mean(), base_rate=y.mean()))
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(os.path.join(args.out, 'calibration.csv'), index=False)

    # ---------------- alarm rules ----------------
    alarm_rows = []
    for m in models:
        for kpers in [1, 2, 3]:
            tp = fp = caught = total = days = 0
            for (sym, k), c in cells.items():
                p, rev = c['preds'][m], c['lab']['rev']
                thr = np.quantile(p['val_event10'], 1 - (c['val']['rev'] <= 10).mean())
                alarm = persistent(p['test_event10'] >= thr, kpers)
                ev = rev <= 10
                tp += (alarm & ev).sum(); fp += (alarm & ~ev).sum(); days += len(rev)
                for t in np.where(rev == 1)[0]:
                    total += 1
                    caught += alarm[max(0, t - 9):t + 1].any()
            alarm_rows.append(dict(model=m, persistence=kpers, precision=tp / max(tp + fp, 1), alarm_rate=(tp + fp) / days,
                                   recall_events=caught / max(total, 1), reversals=total, false_alarm_days_per_year=fp / days * 252))
    alarm_df = pd.DataFrame(alarm_rows)
    alarm_df.to_csv(os.path.join(args.out, 'alarms.csv'), index=False)

    # ---------------- peak_exit per index, with random-exit benchmark ----------------
    bt_rows = []
    for m in models:
        for sym in symbols:
            for kpers in [1, 2]:
                r_strat, r_bh, mkt, held = [], [], [], []
                for k in sorted(k for s, k in cells if s == sym):
                    c = cells[(sym, k)]
                    p, lab = c['preds'][m], c['lab']
                    thr = np.quantile(p['val_event10'], 1 - (c['val']['rev'] <= 10).mean())
                    peak = (p['test_trend'][:, 0] < 0.5) & persistent(p['test_event10'] >= thr, kpers)
                    pos = (~peak).astype(float)[:-2]
                    ret = lab['open'][2:] / lab['open'][1:-1] - 1
                    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
                    r_strat.append(pos * ret - turn * COST); r_bh.append(ret); mkt.append(ret); held.append(pos)
                rs, rb, mk, hd = map(np.concatenate, (r_strat, r_bh, mkt, held))
                if not np.isfinite(mk).all():
                    continue
                # random exits with the same flat spells
                spells, i, t = [], 0, len(hd)
                while i < t:
                    if hd[i] == 0:
                        j = i
                        while j < t and hd[j] == 0:
                            j += 1
                        spells.append(j - i); i = j
                    else:
                        i += 1
                rand = []
                for _ in range(300):
                    rp = np.ones(t)
                    for L in spells:
                        s0 = RNG.integers(0, t - L + 1); rp[s0:s0 + L] = 0
                    tr = np.abs(np.diff(np.concatenate([[0.0], rp])))
                    rr = rp * mk - tr * COST
                    rand.append(rr.mean() / rr.std() * np.sqrt(252))
                ps, pb = perf(rs), perf(rb)
                bt_rows.append(dict(model=m, symbol=sym, persistence=kpers, sharpe=ps['sharpe'], sharpe_bh=pb['sharpe'],
                                    cagr=ps['cagr'], cagr_bh=pb['cagr'], max_dd=ps['max_dd'], max_dd_bh=pb['max_dd'],
                                    flat_share=1 - hd.mean(), beats_random=np.mean(np.array(rand) < ps['sharpe'])))
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(os.path.join(args.out, 'peak_exit_by_index.csv'), index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print('== event metrics (mean over symbol x fold cells)'); print(summ.round(4).to_string())
    print('== pooled stratified bootstrap'); print(tests.round(4).to_string())
    print('== calibration (event10, pooled)'); print(cal_df.round(4).to_string())
    print('== alarm rules (event10, pooled)'); print(alarm_df.round(4).to_string())
    print('== peak_exit by index'); print(bt_df.round(3).to_string())


if __name__ == '__main__':
    main()
