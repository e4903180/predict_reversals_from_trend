"""Fixed one-share backtests (long-short and long-only) from saved signals, vs random timing and buy & hold."""
import json, io, sys, glob, os, numpy as np, pandas as pd
exp_dir, csv = sys.argv[1], sys.argv[2]
px = pd.read_csv(csv, index_col=0, parse_dates=True)['Close']
FEE = 0.000008
def sim(orders, prices, long_only):
    pos, cash, val = 0, 0.0, 0.0
    for o, p in zip(orders, prices):
        target = pos
        if o == 'Buy': target = 1
        elif o == 'Sell': target = 0 if long_only else -1
        trade = target - pos
        if trade:
            cash -= trade * p + (abs(trade) * FEE * p if trade < 0 else 0)  # fee on sells, as in the evaluator
            pos = target
        val = cash + pos * p
    return val
rng = np.random.default_rng(0)
rows = []
for exp in sorted(os.listdir(exp_dir)):
    for split, f in [('val', 'val_summary'), ('test', 'summary')]:
        path = f'{exp_dir}/{exp}/reports/{f}.json'
        if not os.path.exists(path): continue
        d = json.load(open(path))
        td = pd.read_json(io.StringIO(d['trade_details']))
        n_days = len(np.array(d['y_data']))
        first = pd.Timestamp(d['y_data'] and td.index.min()) if len(td) else None
        orders, prices = td['Order'].values, td['Close price'].values
        res = dict(experiment=exp, split=split, n_signals=len(td), n_buy=int((orders == 'Buy').sum()),
                   evaluator_longshort=float(td['Profit'].iloc[-1]) if len(td) else 0.0)
        span = px.loc[td.index.min():td.index.max()] if len(td) else px.iloc[:0]
        for mode, lo in [('longshort', False), ('longonly', True)]:
            mine = sim(orders, prices, lo) if len(td) else 0.0
            sims = []
            for _ in range(2000):
                idx = np.sort(rng.choice(len(span), size=len(td), replace=False))
                sims.append(sim(rng.permutation(orders), span.values[idx], lo))
            sims = np.array(sims) if len(td) else np.zeros(1)
            res.update({f'{mode}_profit': mine, f'{mode}_random_mean': sims.mean(), f'{mode}_random_p95': np.percentile(sims, 95),
                        f'{mode}_pct_rank_vs_random': (sims < mine).mean()})
        res['buy_hold_same_span'] = float(span.iloc[-1] - span.iloc[0]) if len(span) else 0.0
        rows.append(res)
r = pd.DataFrame(rows)
r.to_csv(f'{exp_dir}/backtest_fixed_position.csv', index=False)
pd.set_option('display.width', 250)
print(r.round(2).to_string())
