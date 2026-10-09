"""Runs a grid of model configurations on the same (cached) dataset and summarises them.

Usage (from the repository root):
    python research/scripts/run_experiments.py <out_dir> [--quick] [--models=GRU,LSTM,...] [--dropout=0.1]
                                               [--with-original] [--lightgbm]

Each experiment writes the usual reports/plots under <out_dir>/<name>/, and the
script appends one row per (experiment, split) to <out_dir>/summary.csv, together
with rule-based baselines (always-uptrend, momentum, MA distance, buy & hold, oracle).
"""
import copy
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, os.getcwd())
import main as main_module  # noqa: E402
from evaluator.evaluator import Evaluator  # noqa: E402
from postprocessor.postprocessor import Postprocessor  # noqa: E402
from preprocessor.preprocessor import Preprocessor  # noqa: E402

OUT_DIR = sys.argv[1]
QUICK = '--quick' in sys.argv

MODELS = next((a.split('=', 1)[1].split(',') for a in sys.argv if a.startswith('--models=')),
              ['GRU', 'LSTM', 'TransformerModel', 'TransformerEncoderPE'])
SEEDS = [42] if QUICK else [42, 1, 2]
DROPOUT = float(next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--dropout=')), 0))
COMMON = {'learning_rate': 1e-4, 'training_epoch_num': 50, 'patience': 10, 'batch_size': 32,
          'threshold': 'auto', 'dropout': DROPOUT}
EXPERIMENTS = []
if '--with-original' in sys.argv:
    EXPERIMENTS.append(('Transformer_orig_lr1e-5_ep10', {'model_type': 'TransformerModel', 'learning_rate': 1e-5,
                                                          'training_epoch_num': 10, 'patience': 100}))
for model_type in MODELS:
    for seed in SEEDS:
        EXPERIMENTS.append((f'{model_type}_s{seed}', {**COMMON, 'model_type': model_type, 'seed': seed}))
if QUICK:
    EXPERIMENTS = [(n, {**o, 'training_epoch_num': 2}) for n, o in EXPERIMENTS]

with open('parameters.json') as f:
    BASE_PARAMS = json.load(f)
BASE_PARAMS['model_params']['TransformerEncoderPE'] = {'d_model': 64, 'num_layers': 2, 'num_heads': 4}

# ---- fetch and preprocess once, reuse for every experiment ----
_cached = Preprocessor(BASE_PARAMS).get_datasets()
Preprocessor.get_datasets = lambda self: _cached
_, _, X_val, y_val, X_test, y_test, _, test_dates, val_dates, target = _cached
close = target['Close']


def rule_scores(dates):
    """Downtrend scores known at the close before each window's first predicted day."""
    first_pred_day = pd.DatetimeIndex([d[0] for d in dates])
    pos = close.index.get_indexer(first_pred_day) - 1
    last_close = close.iloc[pos].values
    momentum = -(last_close / close.shift(5).iloc[pos].values - 1)
    ma_distance = -(last_close / close.rolling(20).mean().iloc[pos].values - 1)
    return {'momentum5': momentum, 'ma20_distance': ma_distance}


def per_step_auc(y_true, score):
    out = []
    for k in range(y_true.shape[1]):
        s = score if score.ndim == 1 else score[:, k]
        out.append(roc_auc_score(y_true[:, k], s) if len(np.unique(y_true[:, k])) > 1 else np.nan)
    return np.array(out)


def oracle_and_bh(params, y_true, dates):
    post = Postprocessor(params)
    sig, idx = post.get_first_trend_reversal_and_idx_signals(torch.tensor(y_true))
    rev, _, _ = post.calculate_reversal_dates_with_signals(sig, idx, dates, target)
    oracle = Evaluator(params).execute_trades(post.get_trade_signals_from_reversal_dates(rev, dates, target), target)
    period = close.loc[dates[0][0]:dates[-1][-1]]
    return float(oracle['Profit'].iloc[-1]), float(period.iloc[-1] - period.iloc[0])


rows = []
baseline_cache = {}
for split, y_true_t, dates in [('val', y_val, val_dates), ('test', y_test, test_dates)]:
    y_true = y_true_t.numpy()
    oracle_profit, bh_profit = oracle_and_bh(BASE_PARAMS, y_true, dates)
    baseline_cache[split] = {'oracle_profit': oracle_profit, 'buy_hold_profit': bh_profit}
    rows.append({'experiment': 'always_uptrend', 'split': split, 'auc_flat': 0.5,
                 'trend_acc': float((y_true == 0).mean()), 'balanced_acc': 0.5})
    for rule, score in rule_scores(dates).items():
        aucs = per_step_auc(y_true, score)
        rows.append({'experiment': f'rule_{rule}', 'split': split,
                     'auc_flat': roc_auc_score(y_true.ravel(), np.repeat(score, y_true.shape[1])),
                     'auc_d1_2': np.nanmean(aucs[:2]), 'auc_d1_5': np.nanmean(aucs[:5]), 'auc_d6_16': np.nanmean(aucs[5:]),
                     **{f'auc_d{k + 1}': a for k, a in enumerate(aucs)}})



def window_features(X):
    """Last day, 5-day mean, 64-day mean and 5-day change of every (window-scaled) feature."""
    X = X.numpy()
    return np.concatenate([X[:, -1], X[:, -5:].mean(1), X.mean(1), X[:, -1] - X[:, -6]], axis=1)


if '--lightgbm' in sys.argv:
    import lightgbm as lgb
    X_train_t, y_train_t = _cached[0], _cached[1]
    F_train, F_val, F_test = window_features(X_train_t), window_features(X_val), window_features(X_test)
    for seed in SEEDS:
        probs = {'val': [], 'test': []}
        for k in range(y_train_t.shape[1]):
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=seed, verbose=-1)
            clf.fit(F_train, y_train_t[:, k].numpy())
            probs['val'].append(clf.predict_proba(F_val)[:, 1])
            probs['test'].append(clf.predict_proba(F_test)[:, 1])
        for split, ys in [('val', y_val), ('test', y_test)]:
            yt, prob = ys.numpy(), np.stack(probs[split], axis=1)
            aucs = per_step_auc(yt, prob)
            rows.append({'experiment': f'LightGBM_s{seed}', 'split': split, 'model': 'LightGBM', 'seed': seed,
                         'auc_flat': roc_auc_score(yt.ravel(), prob.ravel()), 'auc_d1_2': np.nanmean(aucs[:2]),
                         'auc_d1_5': np.nanmean(aucs[:5]), 'auc_d6_16': np.nanmean(aucs[5:]),
                         **{f'auc_d{k + 1}': a for k, a in enumerate(aucs)}})
        print(f'done LightGBM_s{seed}', flush=True)

for name, overrides in EXPERIMENTS:
    params = copy.deepcopy(BASE_PARAMS)
    params.update(overrides)
    exp_dir = os.path.join(OUT_DIR, name)
    for sub in ['models', 'plots', 'reports']:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    params['save_path'] = {k: v.replace('outputs/', exp_dir + '/') for k, v in params['save_path'].items()}
    start = time.time()
    val_result, test_result = main_module.ReversePrediction().run(params)
    elapsed = time.time() - start
    for split, res in [('val', val_result), ('test', test_result)]:
        y_true = np.array(res['y_data'])
        y_prob = np.array(res['y_preds'])
        thr = res['usingData']['threshold_used']
        aucs = per_step_auc(y_true, y_prob)
        rd = pd.read_json(io.StringIO(res['reverse_difference']))
        td = pd.read_json(io.StringIO(res['trade_details']))
        lo = pd.read_json(io.StringIO(res['trade_details_long_only']))
        rows.append({
            'experiment': name, 'split': split, 'model': params['model_type'], 'seed': params.get('seed', 42),
            'lr': params['learning_rate'], 'epochs': params['training_epoch_num'],
            'threshold': thr,
            'auc_flat': roc_auc_score(y_true.ravel(), y_prob.ravel()),
            'auc_d1_2': np.nanmean(aucs[:2]), 'auc_d1_5': np.nanmean(aucs[:5]), 'auc_d6_16': np.nanmean(aucs[5:]),
            'trend_acc': float(((y_prob > thr) == y_true).mean()),
            'balanced_acc': balanced_accuracy_score(y_true.ravel(), (y_prob > thr).ravel()),
            'pred_down_share': float((y_prob > thr).mean()), 'true_down_share': float(y_true.mean()),
            'true_reversals': len(rd), 'type_correct': int(rd['reverse_signal_correct'].sum()),
            'in_range_5d': int(res['reverse_in_range_num']),
            'n_signals': len(td), 'n_buy': int((td['Order'] == 'Buy').sum()) if len(td) else 0,
            'longshort_profit': float(td['Profit'].iloc[-1]) if len(td) else 0.0,
            'longonly_profit': float(lo['Profit'].iloc[-1]) if len(lo) else 0.0,
            **baseline_cache[split], 'seconds': elapsed,
            **{f'auc_d{k + 1}': a for k, a in enumerate(aucs)},
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    print(f'done {name} in {elapsed:.0f}s', flush=True)

print(pd.DataFrame(rows)[['experiment', 'split', 'auc_flat', 'auc_d1_2', 'auc_d6_16', 'trend_acc', 'balanced_acc',
                          'in_range_5d', 'longshort_profit']].round(3).to_string())
