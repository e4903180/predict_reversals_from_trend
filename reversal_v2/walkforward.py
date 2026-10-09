"""Walk-forward training of the v2 models; saves daily validation/test predictions.

Usage (from the repository root):
    python -m reversal_v2.walkforward --out research/results/v2/run1 \
        --models logreg,lgbm,GRU-hazard,GRU-bce16 --folds 0-7 --seeds 42,1,2

Folds: two-year test blocks 2008-09, ..., 2022-23 on ^GSPC. For each fold the model is
trained on all four indices up to (validation start - purge), validated (early stopping,
thresholds) on the two years before the test block, and tested on the block. A purge of
60 calendar days separates the sets, because a label at day t looks up to 16 + 20 trading
days ahead.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from reversal_v2 import data as D
from reversal_v2.models import ReversalNet

HORIZON = 16
LOOK_BACK = 64
PURGE = pd.Timedelta(days=60)
EVENT_M = (5, 10)
FOLDS = [(pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y + 1}-12-31')) for y in range(2008, 2024, 2)]
FIRST_DATE = pd.Timestamp('1993-01-01')  # after the 200-day warm-up of the 1992 series


FEATURE_GROUPS = {
    'confirmed': ['conf_type', 'conf_age', 'conf_move'],
    'macro': ['vix', 'vix_ma20', 'vix_chg5', 'tnx', 'tnx_chg20', 'irx_chg20', 'term_spread'],
    'volume': ['volu', 'mfi'],
}


def load_all(indices=None, drop_groups=()):
    indices = indices or D.INDICES
    frames = {s: D.build_index_frame(s) for s in indices}
    drop = {c for g in drop_groups for c in FEATURE_GROUPS[g]}
    cols = [c for c in D.feature_columns(frames['GSPC']) if c not in drop]
    out = {}
    for s, f in frames.items():
        f = f.loc[f.index >= FIRST_DATE]
        Y, first, rev, valid = D.make_targets(f['trend'], HORIZON)
        out[s] = dict(dates=f.index, X=f[cols].values.astype(np.float64), Y=Y, first=first, rev=rev,
                      valid=valid & ~np.isnan(first), close=f['close'].values, open=f['open'].values,
                      ma20=f['ma20'].values, conf_age=f['conf_age'].values, conf_type=f['conf_type'].values)
    return out, cols


def fold_positions(d, start, end, look_back=LOOK_BACK):
    """Positions t with a full look-back window, valid labels and date in [start, end]."""
    pos = np.arange(len(d['dates']))
    m = (pos >= look_back - 1) & d['valid'] & (d['dates'] >= start) & (d['dates'] <= end)
    return pos[m]


def split_fold(all_data, test_start, test_end):
    val_start = test_start - pd.DateOffset(years=2)
    sets = {'train': {}, 'val': {}, 'test': {}}
    for s, d in all_data.items():
        sets['train'][s] = fold_positions(d, FIRST_DATE, val_start - PURGE)
    g = all_data['GSPC']
    sets['val']['GSPC'] = fold_positions(g, val_start, test_start - PURGE)
    sets['test']['GSPC'] = fold_positions(g, test_start, test_end)
    return sets


def standardise(all_data, train_sets):
    rows = np.concatenate([all_data[s]['X'][p] for s, p in train_sets.items()])
    mu, sd = np.nanmean(rows, 0), np.nanstd(rows, 0) + 1e-8
    Xs = {}
    for s, d in all_data.items():
        z = np.clip((d['X'] - mu) / sd, -5, 5)
        Xs[s] = np.nan_to_num(z).astype(np.float32)
    return Xs


def gather(all_data, Xs, part):
    """Windows (as a lazy view + positions) and targets for one split."""
    items = []
    for s, pos in part.items():
        d = all_data[s]
        win = sliding_window_view(Xs[s], LOOK_BACK, axis=0)  # (n-L+1, F, L), no copy
        items.append(dict(sym=s, pos=pos, win=win, Y=d['Y'][pos], first=d['first'][pos], rev=d['rev'][pos],
                          dates=d['dates'][pos], last=Xs[s][pos], lag5=Xs[s][pos - 5], lag20=Xs[s][pos - 20]))
    return items


def windows(item, idx=None):
    pos = item['pos'] if idx is None else item['pos'][idx]
    return np.ascontiguousarray(item['win'][pos - LOOK_BACK + 1].transpose(0, 2, 1))


def tabular(items):
    X = np.concatenate([np.hstack([it['last'], it['lag5'], it['lag20']]) for it in items])
    cat = lambda k: np.concatenate([it[k] for it in items])
    return X, cat('Y'), cat('first'), cat('rev')


# ------------------------------------------------------------------ neural nets
def parse_spec(spec):
    """'GRU-hazard-h32-lr3e-4-do0.3-wd1e-3' -> ('GRU', 'hazard', {'hidden': 32}, lr, weight decay)."""
    parts = spec.split('-')
    encoder, head, enc_kw, lr, wd = parts[0], parts[1], {}, 1e-3, 1e-4
    i = 2
    while i < len(parts):
        tok = parts[i]
        if tok[-1] == 'e' and i + 1 < len(parts):  # re-join scientific notation split by '-'
            tok, i = tok + '-' + parts[i + 1], i + 1
        if tok.startswith('lr'):
            lr = float(tok[2:])
        elif tok.startswith('wd'):
            wd = float(tok[2:])
        elif tok.startswith('do'):
            enc_kw['dropout'] = float(tok[2:])
        elif tok.startswith('h'):
            enc_kw['hidden'] = int(tok[1:])
        i += 1
    return encoder, head, enc_kw, lr, wd


def train_nn(spec, seed, train_items, val_items, n_features, max_epochs=40, patience=5, batch=256):
    encoder, head, enc_kw, lr, wd = parse_spec(spec)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ReversalNet(encoder, n_features, HORIZON, head, **enc_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # flatten (item, row) pairs for shuffling
    index = np.concatenate([np.stack([np.full(len(it['pos']), i), np.arange(len(it['pos']))], 1)
                            for i, it in enumerate(train_items)])
    Xv = torch.tensor(np.concatenate([windows(it) for it in val_items]))
    _, Yv, Fv, Rv = tabular(val_items)
    Yv, Fv, Rv = map(torch.tensor, (Yv.astype(np.float32), Fv.astype(np.float32), Rv))
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(len(index))
        total = 0.0
        for b in range(0, len(perm), batch):
            sel = index[perm[b:b + batch]]
            xb, yb, fb, rb = [], [], [], []
            for i in np.unique(sel[:, 0]):
                rows = sel[sel[:, 0] == i, 1]
                it = train_items[i]
                xb.append(windows(it, rows)); yb.append(it['Y'][rows]); fb.append(it['first'][rows]); rb.append(it['rev'][rows])
            xb = torch.tensor(np.concatenate(xb))
            yb = torch.tensor(np.concatenate(yb).astype(np.float32))
            fb = torch.tensor(np.concatenate(fb).astype(np.float32))
            rb = torch.tensor(np.concatenate(rb))
            opt.zero_grad()
            loss = model.loss(model(xb), yb, fb, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(xb)
        model.eval()
        with torch.no_grad():
            vloss = float(model.loss(model(Xv), Yv, Fv, Rv))
        history.append((epoch + 1, total / len(index), vloss))
        if vloss < best - 1e-4:
            best, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def nn_predict(model, items):
    X = torch.tensor(np.concatenate([windows(it) for it in items]))
    with torch.no_grad():
        logits = model(X)
        trend = model.trend_probs(logits).numpy()
        events = {m: model.event_probs(logits, m).numpy() for m in EVENT_M}
    return trend, events


# ------------------------------------------------------------------ tabular models
def fit_tabular(kind, seed, Xtr, Ytr, Ftr, Rtr):
    targets = {f'y{k}': Ytr[:, k] for k in range(HORIZON)}
    targets.update({f'e{m}': (Rtr <= m).astype(float) for m in EVENT_M})
    models = {}
    for name, y in targets.items():
        if kind == 'logreg':
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=0.05, max_iter=2000)
        else:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                     random_state=seed, verbose=-1)
        clf.fit(Xtr, y.astype(int))
        models[name] = clf
    return models


def tabular_predict(models, X):
    trend = np.stack([models[f'y{k}'].predict_proba(X)[:, 1] for k in range(HORIZON)], 1)
    return trend, {m: models[f'e{m}'].predict_proba(X)[:, 1] for m in EVENT_M}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='logreg,lgbm,GRU-hazard,GRU-bce16')
    ap.add_argument('--folds', default='0-7')
    ap.add_argument('--seeds', default='42,1,2')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--train-indices', default=','.join(D.INDICES), help='indices used for training (GSPC always used for val/test)')
    ap.add_argument('--drop-features', default='', help='comma-separated groups from FEATURE_GROUPS to leave out')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    a, b = (args.folds.split('-') + [None])[:2]
    folds = range(int(a), int(b or a) + 1)
    seeds = [int(s) for s in args.seeds.split(',')]
    indices = args.train_indices.split(',')
    assert 'GSPC' in indices
    all_data, cols = load_all(indices, [g for g in args.drop_features.split(',') if g])
    os.makedirs(args.out, exist_ok=True)
    json.dump({'features': cols, 'train_indices': indices, 'models': args.models, 'horizon': HORIZON, 'look_back': LOOK_BACK, 'folds': [[str(s.date()), str(e.date())] for s, e in FOLDS]},
              open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
    for k in folds:
        test_start, test_end = FOLDS[k]
        sets = split_fold(all_data, test_start, test_end)
        Xs = standardise(all_data, sets['train'])
        items = {name: gather(all_data, Xs, part) for name, part in sets.items()}
        fold_dir = os.path.join(args.out, f'fold{k}')
        os.makedirs(fold_dir, exist_ok=True)
        meta = {split: dict(dates=np.concatenate([it['dates'] for it in items[split]]).astype('datetime64[D]').astype(str),
                            Y=tabular(items[split])[1], first=tabular(items[split])[2], rev=tabular(items[split])[3],
                            close=all_data['GSPC']['close'][items[split][0]['pos']],
                            open=all_data['GSPC']['open'][items[split][0]['pos']],
                            ma20=all_data['GSPC']['ma20'][items[split][0]['pos']],
                            conf_age=all_data['GSPC']['conf_age'][items[split][0]['pos']],
                            conf_type=all_data['GSPC']['conf_type'][items[split][0]['pos']])
                for split in ('val', 'test')}
        np.savez_compressed(os.path.join(fold_dir, 'labels.npz'), **{f'{s}_{k2}': v for s, d in meta.items() for k2, v in d.items()})
        n_train = sum(len(it['pos']) for it in items['train'])
        print(f'fold {k} test {test_start.date()}..{test_end.date()} train={n_train} val={len(meta["val"]["first"])} test={len(meta["test"]["first"])}', flush=True)
        for spec in args.models.split(','):
            for seed in (seeds if spec not in ('logreg', 'lgbm') else seeds[:1]):
                path = os.path.join(fold_dir, f'{spec}_s{seed}.npz')
                if os.path.exists(path):
                    continue
                t0 = time.time()
                preds, info = {}, {}
                if spec in ('logreg', 'lgbm'):
                    Xtr, Ytr, Ftr, Rtr = tabular(items['train'])
                    models = fit_tabular(spec, seed, Xtr, Ytr, Ftr, Rtr)
                    for split in ('val', 'test'):
                        trend, ev = tabular_predict(models, tabular(items[split])[0])
                        preds[f'{split}_trend'] = trend
                        preds.update({f'{split}_event{m}': v for m, v in ev.items()})
                else:
                    model, history = train_nn(spec, seed, items['train'], items['val'], len(cols))
                    info['history'] = history
                    for split in ('val', 'test'):
                        trend, ev = nn_predict(model, items[split])
                        preds[f'{split}_trend'] = trend
                        preds.update({f'{split}_event{m}': v for m, v in ev.items()})
                np.savez_compressed(path, **preds, info=json.dumps(info))
                print(f'  {spec} seed {seed}: {time.time() - t0:.0f}s'
                      + (f" epochs={len(info['history'])} best_val={min(h[2] for h in info['history']):.4f}" if 'history' in info else ''),
                      flush=True)


if __name__ == '__main__':
    main()
