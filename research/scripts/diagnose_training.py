"""Sanity checks that the models actually learn something.

Usage (from the repository root):
    python research/scripts/diagnose_training.py <out_dir>

Checks:
  1. input data: NaN/inf, constant columns inside windows, label balance
  2. reference loss: BCE of predicting the per-step training prior
  3. learning curve: train/val/test loss and AUC per epoch (no early stopping), grad norm, weight drift
  4. capacity: can the model memorise a small subset?
  5. shuffled labels: does AUC drop to 0.5 when the labels are permuted?
  6. how much of the model's output is explained by the MA20-distance rule
"""
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.getcwd())
from model.modelFactory import ModelFactory  # noqa: E402
from preprocessor.preprocessor import Preprocessor  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
params = json.load(open('parameters.json'))
params['model_params']['TransformerEncoderPE'] = {'d_model': 64, 'num_layers': 2, 'num_heads': 4}
cache = os.path.join(OUT, 'datasets.pt')
if os.path.exists(cache):
    data = torch.load(cache, weights_only=False)
else:
    data = Preprocessor(params).get_datasets()
    torch.save(data, cache)
X_train, y_train, X_val, y_val, X_test, y_test, train_dates, test_dates, val_dates, target = data
report = {}


def log(key, value):
    report[key] = value
    print(f'{key}: {value}', flush=True)


# ---------- 1. input data ----------
for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
    const_share = float((X.amax(dim=1) == X.amin(dim=1)).float().mean())
    log(f'data_{name}', {'shape': list(X.shape), 'nan': int(torch.isnan(X).sum()), 'inf': int(torch.isinf(X).sum()),
                         'x_min': float(X.min()), 'x_max': float(X.max()),
                         'constant_feature_windows_share': round(const_share, 4),
                         'down_share': round(float(y.mean()), 4),
                         'down_share_day1_vs_day16': [round(float(y[:, 0].mean()), 4), round(float(y[:, -1].mean()), 4)]})
const_by_feature = (X_train.amax(dim=1) == X_train.amin(dim=1)).float().mean(dim=0)
log('constant_window_share_by_feature', {c: round(float(v), 3) for c, v in zip(params['feature_cols'], const_by_feature) if v > 0})

# ---------- 2. reference loss ----------
bce = nn.BCELoss()
prior = y_train.mean(dim=0, keepdim=True).clamp(1e-6, 1 - 1e-6)
ref = {name: float(bce(prior.expand_as(y), y)) for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]}
log('prior_bce', ref)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = float(nn.BCEWithLogitsLoss()(logits, y))
        prob = torch.sigmoid(logits).numpy()
    yt = y.numpy()
    return loss, roc_auc_score(yt.ravel(), prob.ravel()), roc_auc_score(yt[:, 0], prob[:, 0]), prob


def train(model_type, X, y, epochs, lr, seed=42, batch_size=32, track=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    p = copy.deepcopy(params)
    p['model_type'] = model_type
    model = ModelFactory.create_model_instance(model_type, p)
    init = [w.detach().clone() for w in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    crit = nn.BCEWithLogitsLoss()
    curve = []
    for ep in range(epochs):
        model.train()
        total, grad_norms = 0.0, []
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))))
            opt.step()
            total += float(loss) * len(xb)
        row = {'epoch': ep + 1, 'train_loss_running': total / len(X), 'grad_norm_mean': float(np.mean(grad_norms)),
               'weight_drift': float(sum((w - w0).norm() ** 2 for w, w0 in zip(model.parameters(), init)) ** 0.5)}
        if track:
            for name, Xs, ys in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
                l, a, a1, _ = evaluate(model, Xs, ys)
                row.update({f'{name}_loss': l, f'{name}_auc': a, f'{name}_auc_d1': a1})
        curve.append(row)
    return model, pd.DataFrame(curve)


# ---------- 3. learning curves ----------
curves = {}
for model_type in ['GRU', 'TransformerEncoderPE']:
    t0 = time.time()
    model, curve = train(model_type, X_train, y_train, epochs=50, lr=1e-4)
    curve.to_csv(os.path.join(OUT, f'curve_{model_type}.csv'), index=False)
    curves[model_type] = (model, curve)
    best = curve.loc[curve.val_loss.idxmin()]
    log(f'curve_{model_type}', {
        'seconds': round(time.time() - t0),
        'epoch1': curve.iloc[0][['train_loss', 'val_loss', 'test_loss', 'val_auc', 'test_auc']].round(4).to_dict(),
        'best_val_epoch': int(best.epoch), 'at_best': best[['train_loss', 'val_loss', 'test_loss', 'val_auc', 'test_auc']].round(4).to_dict(),
        'epoch50': curve.iloc[-1][['train_loss', 'val_loss', 'test_loss', 'train_auc', 'val_auc', 'test_auc']].round(4).to_dict(),
        'grad_norm_first_last': [round(curve.grad_norm_mean.iloc[0], 4), round(curve.grad_norm_mean.iloc[-1], 4)],
        'weight_drift_last': round(curve.weight_drift.iloc[-1], 3)})

# ---------- 4. capacity: memorise 256 samples ----------
idx = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(0))[:256]
for model_type in ['GRU', 'TransformerEncoderPE']:
    m, c = train(model_type, X_train[idx], y_train[idx], epochs=300, lr=1e-3, track=False)
    l, a, _, _ = evaluate(m, X_train[idx], y_train[idx])
    log(f'memorise_256_{model_type}', {'final_train_loss': round(l, 4), 'train_auc': round(a, 4),
                                       'prior_bce_on_subset': round(float(bce(y_train[idx].mean(0, keepdim=True).clamp(1e-6, 1 - 1e-6).expand(256, -1), y_train[idx])), 4)})

# ---------- 5. shuffled labels ----------
perm = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(1))
m, c = train('GRU', X_train, y_train[perm], epochs=20, lr=1e-4, track=False)
log('shuffled_labels_GRU', {k: round(evaluate(m, Xs, ys)[1], 4) for k, Xs, ys in [('train_auc_vs_true', X_train, y_train), ('val_auc', X_val, y_val), ('test_auc', X_test, y_test)]})

# ---------- 6. relation to the MA20-distance rule ----------
close = target['Close']
for split, Xs, ys, dates in [('val', X_val, y_val, val_dates), ('test', X_test, y_test, test_dates)]:
    pos = close.index.get_indexer(pd.DatetimeIndex([d[0] for d in dates])) - 1
    rule = -(close.iloc[pos].values / close.rolling(20).mean().iloc[pos].values - 1)
    for model_type, (model, _) in curves.items():
        _, _, _, prob = evaluate(model, Xs, ys)
        # does the model add anything on top of the rule? logistic regression with and without model output
        from sklearn.linear_model import LogisticRegression
        yt = ys.numpy()[:, 0]
        both = np.c_[rule, prob[:, 0]]
        auc_rule = roc_auc_score(yt, rule)
        auc_model = roc_auc_score(yt, prob[:, 0])
        log(f'rule_vs_{model_type}_{split}_day1', {
            'spearman_model_vs_rule': round(spearmanr(prob[:, 0], rule).correlation, 3),
            'auc_rule': round(auc_rule, 4), 'auc_model': round(auc_model, 4)})

json.dump(report, open(os.path.join(OUT, 'diagnostics.json'), 'w'), indent=1, default=str)
