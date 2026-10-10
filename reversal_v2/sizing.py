"""Risk-sizing with the reversal probability instead of all-or-nothing exits.

Usage: python -m reversal_v2.sizing <run_root> [--symbols GSPC,IXIC,DJI,RUT] [--models GRU-hazard,ENS-hazard]

Strategies (signal at the close, trade at the next open, 5 bp per unit traded):
  half_exit : hold 1.0, but 0.5 on peak-alarm days (current trend up and event10 >= validation threshold)
  graded    : exposure 1 - r when the current trend is up, where r in [0, 1] is the percentile of event10
              within the validation distribution above the threshold (linearly from 1 at thr to 0 at the max)
Each is compared with buy & hold and with 300 random placements of the same exposure path.
"""
import argparse
import os

import numpy as np
import pandas as pd

from reversal_v2.analyze import COST, load_run, perf

RNG = np.random.default_rng(0)


def sharpe(r):
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_root')
    ap.add_argument('--symbols', default='GSPC,IXIC,DJI,RUT')
    ap.add_argument('--models', default='GRU-hazard,ENS-hazard')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows = []
    for sym in args.symbols.split(','):
        folds = load_run(args.run_root if sym == 'GSPC' else os.path.join(args.run_root, sym))
        for f in folds:
            members = ['GRU-hazard', 'TCN-hazard', 'MLP-hazard']
            if all(m in f['preds'] for m in members):
                f['preds']['ENS-hazard'] = {k: np.mean([f['preds'][m][k] for m in members], 0) for k in f['preds'][members[0]]}
        for m in args.models.split(','):
            paths = {'half_exit': [], 'graded': []}
            mkt = []
            for f in folds:
                lab = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('test_')}
                val = {k.split('_', 1)[1]: v for k, v in f['labels'].items() if k.startswith('val_')}
                p = f['preds'][m]
                q = 1 - (val['rev'] <= 10).mean()
                thr = np.quantile(p['val_event10'], q)
                top = p['val_event10'].max()
                up = p['test_trend'][:, 0] < 0.5
                e = p['test_event10']
                paths['half_exit'].append(np.where(up & (e >= thr), 0.5, 1.0)[:-2])
                paths['graded'].append(np.where(up, 1 - np.clip((e - thr) / max(top - thr, 1e-9), 0, 1), 1.0)[:-2])
                mkt.append(lab['open'][2:] / lab['open'][1:-1] - 1)
            mk = np.concatenate(mkt)
            pb = perf(mk)
            for name, parts in paths.items():
                pos = np.concatenate(parts)
                turn = np.abs(np.diff(np.concatenate([[1.0], pos])))
                r = pos * mk - turn * COST
                ps = perf(r)
                rand = []
                for _ in range(300):
                    # same exposure values, shifted to random positions in blocks of 20 days (keeps their clustering)
                    blocks = [pos[i:i + 20] for i in range(0, len(pos), 20)]
                    rp = np.concatenate([blocks[j] for j in RNG.permutation(len(blocks))])[:len(pos)]
                    tr = np.abs(np.diff(np.concatenate([[1.0], rp])))
                    rand.append(sharpe(rp * mk - tr * COST))
                rows.append(dict(symbol=sym, model=m, strategy=name, avg_exposure=pos.mean(), sharpe=ps['sharpe'], sharpe_bh=pb['sharpe'],
                                 cagr=ps['cagr'], cagr_bh=pb['cagr'], vol=ps['vol'], vol_bh=pb['vol'], max_dd=ps['max_dd'], max_dd_bh=pb['max_dd'],
                                 beats_random=np.mean(np.array(rand) < ps['sharpe'])))
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    pd.set_option('display.width', 250)
    print(df.round(3).to_string())


if __name__ == '__main__':
    main()
