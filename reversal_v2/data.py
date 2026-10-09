"""Data loading, causal features and labels for the v2 reversal study.

Every feature at day t only uses information available at the close of day t.
Labels follow the original study: a trend label from local extrema of the close
(order 20 days), and for each day t the trend over the next `horizon` days.
"""
import os

import numpy as np
import pandas as pd
import talib
from scipy.signal import argrelextrema

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'research', 'data', 'raw_v2')
INDICES = ['GSPC', 'IXIC', 'DJI', 'RUT']
MACRO = ['VIX', 'IRX', 'FVX', 'TNX']


def load_raw(symbol):
    """Loads one cached Yahoo series."""
    df = pd.read_csv(os.path.join(RAW_DIR, f'{symbol}.csv'), index_col=0, parse_dates=True)
    return df[~df.index.duplicated()].sort_index()


# ---------------------------------------------------------------- labels
def trend_labels(close, order=20):
    """Trend label (0 = up, 1 = down) between alternating local extrema.

    Same rule as `preprocessor.featureFactory.IndicatorTrend.calculate_trend_LocalExtrema`:
    extrema are points that are the max/min of the surrounding `order` days on each side;
    consecutive extrema of the same type keep only the most extreme one; days from a max
    to the next min are down-trend (1), from a min to the next max up-trend (0).
    """
    values = close.values
    max_idx = argrelextrema(values, np.greater_equal, order=order)[0]
    min_idx = argrelextrema(values, np.less_equal, order=order)[0]
    events = sorted([(i, 'max') for i in max_idx] + [(i, 'min') for i in min_idx])
    filtered = []
    for i, kind in events:
        if filtered and filtered[-1][1] == kind:
            j = filtered[-1][0]
            better = values[i] > values[j] if kind == 'max' else values[i] < values[j]
            if better:
                filtered[-1] = (i, kind)
        elif filtered and filtered[-1][0] == i:
            continue  # flat day counted as both max and min
        else:
            filtered.append((i, kind))
    trend = np.full(len(values), np.nan)
    for (i0, k0), (i1, _) in zip(filtered[:-1], filtered[1:]):
        trend[i0:i1] = 1.0 if k0 == 'max' else 0.0
    trend = pd.Series(trend, index=close.index).ffill()
    return trend, filtered


def confirmed_trend_features(close, order=20):
    """Causal version of the trend label: an extremum at day j is only known at day j + order.

    Returns, for every day t, the type of the last *confirmed* extremum (+1 max, -1 min),
    its age in days, and the move since that extremum.
    """
    values = close.values
    n = len(values)
    roll_max = close.rolling(2 * order + 1, center=True).max().values
    roll_min = close.rolling(2 * order + 1, center=True).min().values
    last_type = np.zeros(n)
    last_pos = np.full(n, -1)
    cur_type, cur_pos = 0, -1
    for t in range(n):
        j = t - order  # the newest day whose extremum status is known at t
        if j >= order:
            is_max = values[j] >= roll_max[j]
            is_min = values[j] <= roll_min[j]
            if is_max and (cur_type != 1 or values[j] > values[cur_pos]):
                cur_type, cur_pos = 1, j
            elif is_min and (cur_type != -1 or values[j] < values[cur_pos]):
                cur_type, cur_pos = -1, j
        last_type[t], last_pos[t] = cur_type, cur_pos
    idx = np.arange(n)
    age = np.where(last_pos >= 0, idx - last_pos, np.nan)
    ref = np.where(last_pos >= 0, values[np.clip(last_pos, 0, None)], np.nan)
    return pd.DataFrame({'conf_type': last_type, 'conf_age': np.log1p(age),
                         'conf_move': np.log(values / ref)}, index=close.index)


# ---------------------------------------------------------------- features
def index_features(df, has_volume=True):
    """Scale-free technical features from Close/High/Low (and Volume)."""
    c, h, l = df['Close'], df['High'], df['Low']
    lc = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in [1, 5, 20, 60]:
        f[f'ret{n}'] = lc.diff(n)
    for n in [5, 20, 60, 200]:
        f[f'ma{n}'] = c / c.rolling(n).mean() - 1
    r1 = lc.diff()
    f['vol20'] = r1.rolling(20).std() * np.sqrt(252)
    f['vol60'] = r1.rolling(60).std() * np.sqrt(252)
    f['vol_ratio'] = f['vol20'] / f['vol60'] - 1
    f['atr'] = talib.ATR(h, l, c, 14) / c
    f['rsi'] = (talib.RSI(c, 14) - 50) / 50
    macd, signal, hist = talib.MACD(c, 12, 26, 9)
    f['macd'] = macd / c
    f['macd_hist'] = hist / c
    k, d = talib.STOCH(h, l, c, 14, 3, 0, 3, 0)
    f['stoch'] = (k - 50) / 50
    f['willr'] = (talib.WILLR(h, l, c, 14) + 50) / 50
    f['cci'] = talib.CCI(h, l, c, 20) / 100
    f['adx'] = talib.ADX(h, l, c, 14) / 100
    f['aroon'] = talib.AROONOSC(h, l, 14) / 100
    upper, mid, lower = talib.BBANDS(c, 20, 2, 2)
    f['bb_pos'] = (c - lower) / (upper - lower) - 0.5
    f['bb_width'] = (upper - lower) / mid
    for n in [20, 60]:
        f[f'dd{n}'] = c / c.rolling(n).max() - 1
        f[f'du{n}'] = c / c.rolling(n).min() - 1
    if has_volume:
        v = df['Volume'].replace(0, np.nan)
        f['volu'] = np.log(v / v.rolling(60).mean())
        f['mfi'] = (talib.MFI(h, l, c, v.fillna(0), 14) - 50) / 50
    f = f.join(confirmed_trend_features(c))
    return f


def macro_features(index):
    """VIX and Treasury-yield features, aligned to `index` (forward-filled)."""
    vix = load_raw('VIX')['Close'].reindex(index).ffill()
    irx = load_raw('IRX')['Close'].reindex(index).ffill()
    tnx = load_raw('TNX')['Close'].reindex(index).ffill()
    f = pd.DataFrame(index=index)
    f['vix'] = np.log(vix)
    f['vix_ma20'] = vix / vix.rolling(20).mean() - 1
    f['vix_chg5'] = np.log(vix).diff(5)
    f['tnx'] = tnx / 10
    f['tnx_chg20'] = tnx.diff(20)
    f['irx_chg20'] = irx.diff(20)
    f['term_spread'] = (tnx - irx) / 10
    return f


def build_index_frame(symbol, order=20):
    """Features + trend label for one index."""
    df = load_raw(symbol)
    df = df[df['Close'] > 0]
    feats = index_features(df, has_volume=True).join(macro_features(df.index))
    trend, _ = trend_labels(df['Close'], order=order)
    out = feats.copy()
    out['trend'] = trend
    out['close'] = df['Close']
    out['open'] = df['Open']
    return out


FEATURES = None  # filled on first build


def feature_columns(frame):
    return [c for c in frame.columns if c not in ('trend', 'close', 'open')]


# ---------------------------------------------------------------- windows
def make_targets(trend, horizon=16):
    """For each day t: the trend over t+1..t+horizon, held constant after the first change.

    Returns
      Y (n, horizon) trend labels, first (n,) current trend y[0], rev (n,) first reversal step
      in 1..horizon-1 or `horizon` when none, and valid (n,) mask (all future labels known).
    """
    vals = trend.values
    n = len(vals)
    Y = np.full((n, horizon), np.nan)
    for k in range(horizon):
        Y[:n - 1 - k, k] = vals[1 + k:]
    valid = ~np.isnan(Y).any(axis=1)
    rev = np.full(n, horizon)
    Yf = Y.copy()
    for t in np.where(valid)[0]:
        row = Y[t]
        change = np.nonzero(row[1:] != row[:-1])[0]
        if len(change):
            r = change[0] + 1
            rev[t] = r
            Yf[t, r:] = row[r]
    return Yf, Yf[:, 0], rev, valid


def make_windows(X, look_back, positions):
    """Stack X[t-look_back+1 : t+1] for each position t."""
    return np.stack([X[t - look_back + 1:t + 1] for t in positions]).astype(np.float32)
