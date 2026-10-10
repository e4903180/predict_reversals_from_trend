"""Metrics, baselines, bootstrap tests and back-tests for a walk-forward run.

Usage (from the repository root):
    python -m reversal_v2.analyze <run_dir> [--out research/results/v2_summary]

Seeds of the same model are averaged (a seed ensemble) before scoring.
Baselines
  rule (trend) : trend score = -(close / MA20 - 1)              (from earlier experiments)
  rule (event) : event probability from the trend's confirmed age (empirical, fitted on all
                ^GSPC days before the validation block; no model)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from reversal_v2 import data as D
from reversal_v2.walkforward import EVENT_M, FOLDS, FIRST_DATE, HORIZON, purge

COST = 0.0005  # 5 bp per unit of position change
RNG = np.random.default_rng(0)


def use_run_label(run_dir):
    """Selects the label definition recorded in the run's config.json (default le20)."""
    cfg = os.path.join(run_dir, 'config.json')
    D.set_label(json.load(open(cfg)).get('label', 'le20') if os.path.exists(cfg) else 'le20')


def load_run(run_dir):
    folds = []
    for k in range(len(FOLDS)):
        fd = os.path.join(run_dir, f'fold{k}')
        if not os.path.exists(os.path.join(fd, 'labels.npz')):
            continue
        lab = dict(np.load(os.path.join(fd, 'labels.npz'), allow_pickle=True))
        preds = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(fd, '*_s*.npz'))):
            name = os.path.basename(f).rsplit('_s', 1)[0]
            preds[name].append(dict(np.load(f, allow_pickle=True)))
        ens = {m: {key: np.mean([p[key] for p in ps], 0) for key in ps[0] if key != 'info'} for m, ps in preds.items()}
        folds.append(dict(k=k, labels=lab, preds=ens, n_seeds={m: len(ps) for m, ps in preds.items()}))
    return folds


def age_hazard_baseline(fold_k, split_labels, symbol='GSPC'):
    """P(reversal within m | confirmed-trend type and age bin), fitted on ^GSPC history before validation."""
    test_start = FOLDS[fold_k][0]
    fit_end = test_start - pd.DateOffset(years=2) - purge()
    f = _frame(symbol)
    f = f.loc[(f.index >= FIRST_DATE) & (f.index <= fit_end)]
    Y, first, rev, valid = D.make_targets(f['trend'], HORIZON)
    bins = np.array([0, 1, 2, 2.5, 3, 3.25, 3.5, 3.75, 4, 4.5, 10])
    key = lambda typ, age: (np.sign(typ) + 1) * 100 + np.digitize(age, bins)
    k_fit = key(f['conf_type'].values, np.nan_to_num(f['conf_age'].values))[valid]
    out = {}
    for m in EVENT_M:
        ev = (rev[valid] <= m).astype(float)
        table = pd.Series(ev).groupby(k_fit).mean()
        k_eval = key(split_labels['conf_type'], np.nan_to_num(split_labels['conf_age']))
        out[m] = pd.Series(k_eval).map(table).fillna(ev.mean()).values
    return out


_FRAMES = {}


def _frame(symbol):
    if symbol not in _FRAMES:
        _FRAMES[symbol] = D.build_index_frame(symbol)
    return _FRAMES[symbol]


def block_bootstrap_auc_diff(y, s_model, s_base, block=20, n=1000):
    """Moving-block bootstrap of AUC(model) - AUC(base) over the pooled test days."""
    t = len(y)
    starts = np.arange(t - block + 1)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in RNG.choice(starts, t // block + 1)])[:t]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, s_model[idx]) - roc_auc_score(yy, s_base[idx]))
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())


def reversal_positions(p_now_down, event, thr):
    """Two strategies that use the reversal (event) probability directly.

    peak_exit       : long, except on days with a peak alarm (current trend up and event >= thr).
    reversal_switch : leave on a peak alarm, stay flat until a valley alarm (current trend down and event >= thr).
    """
    peak = (p_now_down < 0.5) & (event >= thr)
    valley = (p_now_down >= 0.5) & (event >= thr)
    exit_ = (~peak).astype(float)
    switch = np.ones(len(event))
    state = 1.0
    for i in range(len(event)):
        if peak[i]:
            state = 0.0
        elif valley[i]:
            state = 1.0
        switch[i] = state
    return exit_, switch


def backtest(p_down, close, open_, ma20, threshold=0.5, p_now_down=None, event=None, event_thr=None):
    """Signal at the close of day t, trade at the open of t+1; returns are open-to-open."""
    ret = open_[2:] / open_[1:-1] - 1  # return earned by the position chosen at the close of t
    signals = {
        'buy_hold': np.ones(len(ret)),
        'ma20_long': (ma20[:-2] > 0).astype(float),
        'model_long': (p_down[:-2] < threshold).astype(float),
        'model_long_short': np.where(p_down[:-2] < threshold, 1.0, -1.0),
    }
    if event is not None:
        exit_, switch = reversal_positions(p_now_down, event, event_thr)
        signals['model_peak_exit'] = exit_[:-2]
        signals['model_reversal_switch'] = switch[:-2]
    out = {}
    for name, pos in signals.items():
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        r = pos * ret - turn * COST
        out[name] = r
    return out


def perf(r):
    eq = np.cumprod(1 + r)
    years = len(r) / 252
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=eq[-1] ** (1 / years) - 1, vol=r.std() * np.sqrt(252),
                sharpe=r.mean() / (r.std() + 1e-12) * np.sqrt(252), max_dd=dd.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--symbol', default='GSPC', help='index the run directory was tested on (for the age baseline)')
    ap.add_argument('--ensemble', action='append', default=[],
                    help='name=modelA+modelB+... : average the (seed-averaged) probabilities of several models')
    ap.add_argument('--extra-run', action='append', default=[],
                    help='prefix=path : also load the models of another run directory, named prefix/<model>')
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    use_run_label(args.run_dir)
    folds = load_run(args.run_dir)
    for extra in args.extra_run:
        prefix, path = extra.split('=', 1)
        other = {f['k']: f for f in load_run(path)}
        for f in folds:
            for m, p in other.get(f['k'], {'preds': {}})['preds'].items():
                f['preds'][f'{prefix}/{m}'] = p
                f['n_seeds'][f'{prefix}/{m}'] = other[f['k']]['n_seeds'][m]
    for spec in args.ensemble:
        name, members = spec.split('=', 1)
        members = members.split('+')
        for f in folds:
            if all(m in f['preds'] for m in members):
                f['preds'][name] = {key: np.mean([f['preds'][m][key] for m in members], 0) for key in f['preds'][members[0]]}
                f['n_seeds'][name] = sum(f['n_seeds'][m] for m in members)
    models = sorted(set.intersection(*[set(f['preds']) for f in folds]))
    rows, pooled = [], defaultdict(list)
    bt = defaultdict(list)
    for f in folds:
        lab = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('test_')}
        val = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('val_')}
        Y, rev = lab['Y'], lab['rev']
        base = age_hazard_baseline(f['k'], lab, args.symbol)
        scores = {'rule': {'trend': np.repeat(-lab['ma20'][:, None], HORIZON, 1), **{f'event{m}': base[m] for m in EVENT_M}}}
        for m in models:
            p = f['preds'][m]
            scores[m] = {'trend': p['test_trend'], **{f'event{e}': p[f'test_event{e}'] for e in EVENT_M}}
            # threshold for alarms: match the validation base rate of the event
            for e in EVENT_M:
                q = 1 - (val['rev'] <= e).mean()
                scores[m][f'thr{e}'] = np.quantile(p[f'val_event{e}'], q)
        for name, s in scores.items():
            row = {'fold': f['k'], 'model': name}
            row['trend_auc'] = roc_auc_score(Y.ravel(), s['trend'].ravel())
            row['trend_auc_d1_2'] = np.mean([roc_auc_score(Y[:, k], s['trend'][:, k]) for k in range(2)])
            row['trend_auc_d6_16'] = np.mean([roc_auc_score(Y[:, k], s['trend'][:, k]) for k in range(5, HORIZON)])
            for e in EVENT_M:
                ev = (rev <= e).astype(int)
                row[f'event{e}_auc'] = roc_auc_score(ev, s[f'event{e}'])
                row[f'event{e}_ap'] = average_precision_score(ev, s[f'event{e}'])
                row[f'event{e}_base_rate'] = ev.mean()
            rows.append(row)
            pooled[name].append(dict(Y=Y, rev=rev, **{k2: v for k2, v in s.items() if not k2.startswith('thr')},
                                     thr={e: s.get(f'thr{e}') for e in EVENT_M}))
        for m in models:
            p = f['preds'][m]
            q = 1 - (val['rev'] <= 10).mean()
            for name, r in backtest(p['test_trend'][:, :5].mean(1), lab['close'], lab['open'], lab['ma20'],
                                    p_now_down=p['test_trend'][:, 0], event=p['test_event10'],
                                    event_thr=np.quantile(p['val_event10'], q)).items():
                bt[(m, name)].append(r)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(out_dir, 'per_fold.csv'), index=False)
    summary = per_fold.groupby('model').mean(numeric_only=True).drop(columns='fold')
    summary['folds_trend_beats_rule'] = per_fold.pivot(index='fold', columns='model', values='trend_auc').apply(
        lambda c: (c > per_fold[per_fold.model == 'rule'].set_index('fold')['trend_auc']).sum())
    summary['folds_event10_beats_age'] = per_fold.pivot(index='fold', columns='model', values='event10_auc').apply(
        lambda c: (c > per_fold[per_fold.model == 'rule'].set_index('fold')['event10_auc']).sum())

    # pooled bootstrap tests
    def cat(name, key):
        return np.concatenate([d[key] for d in pooled[name]])
    tests = []
    for m in models:
        Yp = cat(m, 'Y')
        for key, label_fn, base in [('trend', None, 'rule'), ('event5', 5, 'rule'), ('event10', 10, 'rule'),
                                    ('event10', 10, 'logreg')]:
            if base == m or base not in pooled:
                continue
            if key == 'trend':
                y = Yp[:, :5].ravel()
                sm, sb = cat(m, 'trend')[:, :5].ravel(), cat(base, 'trend')[:, :5].ravel()
                blk = 20 * 5
                label = 'trend d1-5'
            else:
                y = (cat(m, 'rev') <= label_fn).astype(int)
                sm, sb = cat(m, key), cat(base, key)
                blk = 20
                label = key
            mean, lo, hi, p = block_bootstrap_auc_diff(y, sm, sb, block=blk, n=500)
            tests.append(dict(model=m, metric=label, versus='age_hazard' if (base == 'rule' and key != 'trend') else base,
                              auc_diff=mean, ci_low=lo, ci_high=hi, p_le_0=p))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(out_dir, 'bootstrap_tests.csv'), index=False)

    # event-level alarms (threshold from validation base rate), pooled
    alarm_rows = []
    for m in models:
        for e in EVENT_M:
            tp = fp = 0
            caught = total = 0
            days = 0
            for d in pooled[m]:
                s, rev = d[f'event{e}'], d['rev']
                alarm = s >= d['thr'][e]
                ev = rev <= e
                tp += (alarm & ev).sum(); fp += (alarm & ~ev).sum(); days += len(s)
                # a true reversal day = first day whose window has the reversal at step 1;
                # it is "caught" if any of the e days before it raised an alarm
                starts = np.where(rev == 1)[0]
                for t in starts:
                    total += 1
                    caught += alarm[max(0, t - e + 1):t + 1].any()
            alarm_rows.append(dict(model=m, horizon=e, precision=tp / max(tp + fp, 1), alarm_rate=(tp + fp) / days,
                                   reversals=total, recall_events=caught / max(total, 1),
                                   false_alarm_days_per_year=fp / days * 252))
    alarms = pd.DataFrame(alarm_rows)
    alarms.to_csv(os.path.join(out_dir, 'alarms.csv'), index=False)

    # back-tests: chain the folds
    perf_rows = []
    for (m, name), parts in bt.items():
        perf_rows.append(dict(model=m, strategy=name, **perf(np.concatenate(parts))))
    perf_df = pd.DataFrame(perf_rows)
    perf_df = perf_df[~perf_df.strategy.isin(['buy_hold', 'ma20_long']) | ~perf_df.duplicated(subset=['strategy'])]
    perf_df.to_csv(os.path.join(out_dir, 'backtest.csv'), index=False)

    summary.to_csv(os.path.join(out_dir, 'summary.csv'))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print('== mean over folds'); print(summary.round(4).to_string())
    print('== bootstrap (pooled test days)'); print(tests.round(4).to_string())
    print('== alarms'); print(alarms.round(4).to_string())
    print('== backtest 2008-2023 (next-open execution, 5 bp)'); print(perf_df.round(4).to_string())
    json.dump({m: f for m, f in folds[0]['n_seeds'].items()}, open(os.path.join(out_dir, 'n_seeds.json'), 'w'))


if __name__ == '__main__':
    main()
