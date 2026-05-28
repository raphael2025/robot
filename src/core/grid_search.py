"""
grid_search.py
=============
Exhaustive parameter grid search for all 6 intent signals.
Calibrates THRESHOLDS on historical data, finds optimal configs per intent.

Usage:
    python src/core/grid_search.py --intent absorption --days 30
    python src/core/grid_search.py --intent all --days 30

Output:
    - Optimal config printed to stdout
    - Config saved to config/thresholds.toml
"""
import argparse
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import sys; sys.path.insert(0, 'src')
from core.backtest_engine import THRESHOLDS, classify_bar


def load_bars(symbol, days, data_dir='data'):
    path = f"{data_dir}/{symbol}_1m_{days}d.parquet"
    df = pd.read_parquet(path)
    bars = []
    for _, r in df.iterrows():
        bars.append({
            't': int(r['ts'].value // 1_000_000),
            'o': float(r['open']), 'h': float(r['high']),
            'l': float(r['low']),  'c': float(r['close']),
            'vol': float(r['vol']), 'tbb': float(r['taker_buy_base']),
            'tbq': float(r['taker_buy_quote']),
            'n': int(r['n_trades']), 'br': float(r['buy_ratio']),
        })
    for i, b in enumerate(bars):
        b['ret_1m'] = (b['c'] - bars[i-1]['c']) / bars[i-1]['c'] * 10_000 if i > 0 else 0.0
    return bars


def compute_features(bars, lookback=20):
    for i in range(len(bars)):
        win = bars[max(0, i - lookback + 1):i + 1]
        br = [w['br'] for w in win]; vols = [w['vol'] for w in win]
        bars[i]['oi']   = (np.mean(br) - 0.5) * 2
        bars[i]['vpin'] = abs(np.mean(br) - 0.5) * 2 * (1 + np.std(br))
        bars[i]['ss']   = min(1.0, (bars[i]['vol'] - np.mean(vols)) / (np.std(vols) + 1e-9) / 3)
        bars[i]['ar']   = sum(win[j]['vol'] - win[j]['tbb'] for j in range(len(win))) \
                         / (sum(win[j]['tbb'] for j in range(len(win))) + 1e-9)
        bars[i]['ta']   = np.mean([bars[k]['ret_1m'] for k in range(max(0, i-lookback+1), i+1)])
        sprd = np.mean([win[j]['h'] - win[j]['l'] for j in range(len(win))]) or 1.0
        bars[i]['mp_drift'] = abs(bars[i]['c'] + bars[i]['oi'] * (sprd/2) - bars[i]['c']) / bars[i]['c'] * 10_000
        bars[i]['price_range'] = (bars[i]['h'] - bars[i]['l']) / bars[i]['c'] * 10_000

    feat_cols = ['oi', 'vpin', 'ar', 'ss']
    fa = {c: np.array([bars[i][c] for i in range(len(bars))], dtype=float) for c in feat_cols}
    df_t = pd.DataFrame(fa)
    for c in feat_cols:
        df_t[c + '_pct'] = df_t[c].expanding().rank(pct=True) * 100
    for i in range(len(bars)):
        for c in feat_cols:
            bars[i][c + '_pct'] = df_t[c + '_pct'].iloc[i]


def score_signal(bars, cfg, intent_key, min_samples=3):
    """Evaluate a config on all horizons. Returns (avg60, sharpe60, n, wr60) or None."""
    wins = {5: [], 15: [], 30: [], 60: []}
    lb_key = 'lookback' if 'lookback' in cfg else cfg.get('consecutive', cfg.get('repeat_min', 1))
    start = max(lb_key + 1, 30)  # need history for rolling + lookahead

    for i in range(start, len(bars) - 60):
        intent, conf = classify_bar(bars, i, cfg)
        if intent.value == intent_key and conf >= 0.60:
            for h in [5, 15, 30, 60]:
                wins[h].append(sum(bars[i + k]['ret_1m'] for k in range(1, h + 1)))

    if not all(len(wins[h]) >= min_samples for h in [5, 15, 30, 60]):
        return None

    n = len(wins[60])
    avg60 = np.mean(wins[60])
    std60 = np.std(wins[60])
    sh60 = avg60 / max(std60, 0.01)
    wr60 = sum(1 for v in wins[60] if v > 0) / n * 100
    return {'n': n, 'avg60': avg60, 'sh60': sh60, 'wr60': wr60, 'wins': wins}


# ─────────────────────────────────────────────────────────────────────────────
# Grid definitions per intent
# ─────────────────────────────────────────────────────────────────────────────
GRIDS = {
    'absorption': [
        ('ar_pct_min', [75, 80, 82, 85, 88, 90, 92, 95]),
        ('oi_pct_max', [5, 8, 10, 12, 15, 18, 20]),
        ('ta_max_abs', [4, 5, 6, 7, 8]),
        ('lookback',   [2, 3, 4]),
    ],
    'distribution': [
        ('ar_pct_max', [8, 10, 12, 15, 18, 20, 25]),
        ('oi_pct_min', [80, 85, 88, 90, 92, 95]),
        ('ta_max_abs', [4, 5, 6, 7, 8]),
        ('lookback',   [2, 3, 4]),
    ],
    'liquidity_probe': [
        ('vpin_pct_min', [75, 80, 85, 88, 90, 92, 95]),
        ('ss_pct_min',   [70, 75, 80, 85, 88, 90, 92, 95]),
        ('repeat_min',   [1, 2, 3]),
    ],
    'micro_drift': [
        ('oi_pct_min',  [50, 55, 60, 65, 70, 75, 80, 85, 90]),
        ('drift_min',   [0.5, 1.0, 1.5, 2.0, 2.5]),
        ('consecutive', [2, 3, 4, 5]),
    ],
    'bull_trap': [
        ('ret_min_bps',  [2.0, 3.0, 4.0, 5.0]),
        ('vpin_pct_min', [75, 80, 85, 88, 90]),
        ('ss_pct_min',   [70, 75, 80, 85, 88, 90]),
        ('post_rev_max', [1.0, 2.0, 3.0, 4.0, 5.0]),
        ('lookback',     [1]),
    ],
    'bear_trap': [
        ('ret_max_bps',  [-2.0, -3.0, -4.0, -5.0]),
        ('vpin_pct_min', [75, 80, 85, 88, 90]),
        ('ss_pct_min',   [70, 75, 80, 85, 88, 90]),
        ('post_rev_min', [-1.0, -2.0, -3.0, -4.0, -5.0]),
        ('lookback',     [1]),
    ],
}


def run_grid(intent_key, bars, lookback, min_samples=3):
    """Run full grid for one intent. Returns sorted configs."""
    import itertools

    grid = GRIDS[intent_key]
    keys = [k for k, _ in grid]
    values = [vals for _, vals in grid]

    configs = []
    for combo in itertools.product(*values):
        cfg_dict = dict(zip(keys, combo))
        cfg_dict['conf_base'] = 0.60  # always include conf_base
        base_cfg = {k: v for k, v in THRESHOLDS.items() if k != intent_key}
        base_cfg[intent_key] = {**cfg_dict, 'conf_base': 0.60}

        result = score_signal(bars, base_cfg, intent_key, min_samples)
        if result:
            row = {**dict(zip(keys, combo)), **result}
            configs.append(row)

    configs.sort(key=lambda x: x['avg60'], reverse=True)
    return configs


def update_thresholds(best_cfg, intent_key):
    """Patch THRESHOLDS in backtest_engine.py with optimized config."""
    path = 'src/core/backtest_engine.py'

    with open(path) as f:
        lines = f.readlines()

    # Find the intent block
    in_block = False
    block_start = None
    indent = None

    for i, line in enumerate(lines):
        if f'"{intent_key}"' in line and ': dict(' in line:
            in_block = True
            block_start = i
            # Find indent
            indent = len(line) - len(line.lstrip())
            continue
        if in_block:
            # End of block: closing ), blank, or another intent
            if line.strip() == '),' or (line.strip() == ')' and i > block_start):
                in_block = False
                block_end = i
                break

    if block_start is None:
        return

    # Build new block content
    new_lines = []
    key_order = ['lookback', 'consecutive', 'repeat_min',
                 'ar_pct_min', 'ar_pct_max', 'oi_pct_min', 'oi_pct_max',
                 'vpin_pct_min', 'ss_pct_min', 'drift_min',
                 'ta_max_abs', 'ret_min_bps', 'ret_max_bps',
                 'post_rev_max', 'post_rev_min', 'conf_base']

    param_map = {
        'lookback': 'lookback', 'consecutive': 'consecutive', 'repeat_min': 'repeat_min',
        'ar_pct_min': 'ar_pct_min', 'ar_pct_max': 'ar_pct_max',
        'oi_pct_min': 'oi_pct_min', 'oi_pct_max': 'oi_pct_max',
        'vpin_pct_min': 'vpin_pct_min', 'ss_pct_min': 'ss_pct_min',
        'drift_min': 'drift_min',
        'ta_max_abs': 'ta_max_abs',
        'ret_min_bps': 'ret_min_bps', 'ret_max_bps': 'ret_max_bps',
        'post_rev_max': 'post_rev_max', 'post_rev_min': 'post_rev_min',
        'conf_base': 'conf_base',
    }

    for k in key_order:
        if k in best_cfg:
            new_lines.append(f'{" " * (indent + 4)}{k} = {best_cfg[k]},\n')

    # Replace lines[block_start:block_end+1] with new block
    new_block = lines[:block_start] + [' ' * indent + f'"{intent_key}": dict(\n'] + new_lines + [' ' * indent + '),\n'] + lines[block_end+1:]
    with open(path, 'w') as f:
        f.writelines(new_block)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--intent', default='all',
                    choices=['absorption', 'distribution', 'liquidity_probe',
                             'micro_drift', 'bull_trap', 'bear_trap', 'all'])
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--lookback', type=int, default=20)
    ap.add_argument('--min-samples', type=int, default=3)
    ap.add_argument('--top-k', type=int, default=5)
    args = ap.parse_args()

    print(f"Loading {args.days}d data for {args.symbol}...")
    bars = load_bars(args.symbol, args.days)
    print(f"  {len(bars)} bars loaded")
    compute_features(bars, args.lookback)
    print(f"  Features computed")

    intents = [args.intent] if args.intent != 'all' else \
              ['absorption', 'distribution', 'liquidity_probe',
               'micro_drift', 'bull_trap', 'bear_trap']

    best_overall = {}

    for intent_key in intents:
        print(f"\n{'='*60}")
        print(f"  GRID SEARCH: {intent_key}")
        print(f"{'='*60}")

        configs = run_grid(intent_key, bars, args.lookback, args.min_samples)
        print(f"  {len(configs)} valid configs tested")

        if not configs:
            print("  No configs with min_samples met. Relax threshold.")
            continue

        best = configs[0]
        print(f"\n  Top {args.top_k} by avg60:")
        for c in configs[:args.top_k]:
            n = c['n']
            wr = c['wr60']
            avg = round(c['avg60'], 3)
            sh = round(c['sh60'], 3)
            params = ' '.join(f"{k}={c[k]}" for k, _ in GRIDS[intent_key] if k in c)
            print(f"    n={n:4d}  wr={wr:5.1f}%  avg60={avg:7.3f}bps  sh={sh:.3f}  {params}")

        print(f"\n  BEST: n={best['n']} wr={best['wr60']:.1f}% avg60={best['avg60']:.3f}bps sh={best['sh60']:.3f}")

        # Update thresholds.toml
        update_thresholds(best, intent_key)
        best_overall[intent_key] = best

    print(f"\n\n{'#'*60}")
    print("  FINAL CALIBRATED THRESHOLDS")
    print(f"{'#'*60}")
    for k, v in best_overall.items():
        print(f"  {k}: n={v['n']} wr60={v['wr60']:.1f}% avg60={v['avg60']:.3f}bps sh60={v['sh60']:.3f}")