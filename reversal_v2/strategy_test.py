"""Significance of the reversal-aware strategies against buy & hold and random exits.

Usage: python -m reversal_v2.strategy_test <run_dir> <model> [--extra-run prefix=path ...]
"""
import argparse
import numpy as np

from reversal_v2.analyze import COST, backtest, load_run, perf


def sharpe(r):
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('model')
    ap.add_argument('--extra-run', action='append', default=[])
    ap.add_argument('--n', type=int, default=2000)
    args = ap.parse_args()
    folds = load_run(args.run_dir)
    for extra in args.extra_run:
        prefix, path = extra.split('=', 1)
        other = {f['k']: f for f in load_run(path)}
        for f in folds:
            for m, p in other[f['k']]['preds'].items():
                f['preds'][f'{prefix}/{m}'] = p
    rets, pos_all, raw = {'buy_hold': [], 'model_peak_exit': [], 'model_reversal_switch': []}, {}, []
    for f in folds:
        lab = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('test_')}
        val = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('val_')}
        p = f['preds'][args.model]
        thr = np.quantile(p['val_event10'], 1 - (val['rev'] <= 10).mean())
        out = backtest(p['test_trend'][:, :5].mean(1), lab['close'], lab['open'], lab['ma20'],
                       p_now_down=p['test_trend'][:, 0], event=p['test_event10'], event_thr=thr)
        for k in rets:
            rets[k].append(out[k])
        raw.append(lab['open'][2:] / lab['open'][1:-1] - 1)
    R = {k: np.concatenate(v) for k, v in rets.items()}
    market = np.concatenate(raw)
    rng = np.random.default_rng(0)
    for strat in ['model_peak_exit', 'model_reversal_switch']:
        r = R[strat]
        # position actually held (1 or 0), recovered from returns where the market moved
        held = np.isclose(r + COST * 0, market, atol=COST + 1e-12) & (market != 0)
        flat_share = 1 - held.mean()
        # 1) block bootstrap of the Sharpe difference vs buy & hold
        t, block = len(r), 20
        starts = np.arange(t - block + 1)
        diffs = []
        for _ in range(args.n):
            idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, t // block + 1)])[:t]
            diffs.append(sharpe(r[idx]) - sharpe(R['buy_hold'][idx]))
        diffs = np.array(diffs)
        # 2) random exits: same number and lengths of flat spells, placed at random
        pos = held.astype(float)
        spells, i = [], 0
        while i < t:
            if pos[i] == 0:
                j = i
                while j < t and pos[j] == 0:
                    j += 1
                spells.append(j - i)
                i = j
            else:
                i += 1
        rand_sharpes = []
        for _ in range(args.n):
            rp = np.ones(t)
            for L in spells:
                s0 = rng.integers(0, t - L + 1)
                rp[s0:s0 + L] = 0
            turn = np.abs(np.diff(np.concatenate([[0.0], rp])))
            rand_sharpes.append(sharpe(rp * market - turn * COST))
        rand_sharpes = np.array(rand_sharpes)
        pm, pb = perf(r), perf(R['buy_hold'])
        print(f'{args.model} {strat}: Sharpe {pm["sharpe"]:.3f} vs B&H {pb["sharpe"]:.3f}; CAGR {pm["cagr"]:.4f} vs {pb["cagr"]:.4f}; '
              f'MDD {pm["max_dd"]:.3f} vs {pb["max_dd"]:.3f}; flat {flat_share:.1%} of days in {len(spells)} spells')
        print(f'   bootstrap Sharpe diff mean {diffs.mean():+.3f} 95% CI [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}] P(diff<=0)={np.mean(diffs <= 0):.3f}')
        print(f'   random exits (same spells): Sharpe mean {rand_sharpes.mean():.3f}, 95th pct {np.percentile(rand_sharpes, 95):.3f}; '
              f'model beats {np.mean(rand_sharpes < pm["sharpe"]):.1%} of random')


if __name__ == '__main__':
    main()
