"""
backtest_engine.py
==================
Efficient market-cycle backtester using Binance 1m klines.

Signal detection philosophy:
  - Use ONLY what Binance 1m kline provides: open/high/low/close/volume/n_trades/tbb/tbq
  - Compute rolling features (lookback 20) to smooth noise
  - Multi-factor scoring: intent fires when COMBINED score exceeds threshold
    (not when a single loose condition is met)
  - Evaluate forward 5m/15m/30m/60m returns vs buy-hold baseline

Key calibrations (from 14-day data analysis):
  OI:      mean=0.04, std=0.48, p95=±0.79  → threshold ±0.65 (extreme only)
  AR:      mean=2.0, median~1.5             → ABSORPTION >5, DISTRIBUTION <0.35
  returns: 73% of bars <5bps, 93% <10bps   → quiet bars = cleaner signals

Usage:
    python backtest_engine.py --symbol BTCUSDT --days 14
    python backtest_engine.py --symbol BTCUSDT --days 30 --lookback 30
"""

import argparse
import time
import requests
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/home/raphael/robot/src")
from core.orderflow import Intent, IntentSignal


# ─────────────────────────────────────────────────────────────
#  Binance REST
# ─────────────────────────────────────────────────────────────

BASE = "https://api.binance.com/api/v3"

def fetch_klines(symbol, interval, start_ts_ms, limit=1000):
    r = requests.get(
        f"{BASE}/klines",
        params={"symbol": symbol, "interval": interval, "startTime": start_ts_ms, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────
#  Bar Features
# ─────────────────────────────────────────────────────────────

def parse_bars(raw_klines):
    bars = []
    for k in raw_klines:
        open_t    = int(k[0])
        close_t   = int(k[6])
        open_     = float(k[1])
        high     = float(k[2])
        low      = float(k[3])
        close    = float(k[4])
        vol      = float(k[5])
        quote_vol= float(k[7])
        n_trades = int(k[8])
        tbb      = float(k[9])   # taker buy base (aggressive buy BTC)
        tbq      = float(k[10])  # taker buy quote

        buy_ratio = tbb / vol if vol > 0 else 0.5
        ret_1m    = (close - open_) / open_ * 10_000 if open_ > 0 else 0.0

        bars.append({
            "t": open_t, "close_t": close_t,
            "open": open_, "high": high, "low": low, "close": close,
            "vol": vol, "quote_vol": quote_vol, "n_trades": n_trades,
            "tbb": tbb, "tbq": tbq,
            "buy_ratio": buy_ratio,
            "ret_1m": ret_1m,
            # rolling features (filled in next step)
            "oi": 0.0, "vpin": 0.0, "ar": 1.0, "ss": 0.0,
            "ta": 0.0, "mp_drift": 0.0, "price_range": 0.0,
            "oi_pct": 0.0, "vpin_pct": 0.0, "ar_pct": 0.0, "ss_pct": 0.0,
        })
    return bars

def parse_bars_from_df(df):
    """Convert a parquet DataFrame into the bar dicts expected by the backtester."""
    bars = []
    for _, row in df.iterrows():
        open_t = int(row["ts"].value / 1_000_000)  # datetime64[ms] → ms
        close_t = int(row["close_time"].value / 1_000_000)
        open_  = float(row["open"])
        high   = float(row["high"])
        low    = float(row["low"])
        close  = float(row["close"])
        vol    = float(row["vol"])
        n_trades = int(row["n_trades"])
        tbb   = float(row["taker_buy_base"])
        tbq   = float(row["taker_buy_quote"])
        buy_ratio = float(row["buy_ratio"])
        ret_1m = (close - open_) / open_ * 10_000 if open_ > 0 else 0.0

        bars.append({
            "t": open_t, "close_t": close_t,
            "open": open_, "high": high, "low": low, "close": close,
            "vol": vol, "n_trades": n_trades,
            "tbb": tbb, "tbq": tbq,
            "buy_ratio": buy_ratio,
            "ret_1m": ret_1m,
            "oi": 0.0, "vpin": 0.0, "ar": 1.0, "ss": 0.0,
            "ta": 0.0, "mp_drift": 0.0, "price_range": 0.0,
            "oi_pct": 0.0, "vpin_pct": 0.0, "ar_pct": 0.0, "ss_pct": 0.0,
        })
    return bars


def compute_rolling(bars, lookback=20):
    """Compute rolling features + expanding percentile ranks (0-100).

    Rolling features use lookback window (smoothed, no lookahead).
    Percentile ranks use expanding window (all history up to current bar).
    """
    n = len(bars)

    # ── Step 1: compute raw rolling features ─────────────────────────────
    for i in range(n):
        win = bars[max(0, i - lookback + 1):i + 1]
        br      = [b["buy_ratio"] for b in win]
        vol_vals= [b["vol"] for b in win]

        oi   = (np.mean(br) - 0.5) * 2
        vpin = abs(np.mean(br) - 0.5) * 2 * (1 + np.std(br))
        pas  = [win[j]["vol"] - win[j]["tbb"] for j in range(len(win))]
        agg  = [win[j]["tbb"] for j in range(len(win))]
        ar   = sum(pas) / (sum(agg) + 1e-9)
        ta   = np.mean([b["ret_1m"] for b in win])
        vm   = np.mean(vol_vals); vs = np.std(vol_vals) or 1.0
        ss   = min(1.0, (bars[i]["vol"] - vm) / vs / 3)
        sprd = np.mean([b["high"] - b["low"] for b in win[-5:]]) or 1.0
        mp   = bars[i]["close"] + oi * (sprd / 2)
        mp_d = abs(mp - bars[i]["close"]) / bars[i]["close"] * 10_000
        pr   = (bars[i]["high"] - bars[i]["low"]) / bars[i]["close"] * 10_000

        bars[i].update(oi=oi, vpin=vpin, ar=ar, ta=ta, ss=ss,
                       mp_drift=mp_d, price_range=pr)

    # ── Step 2: expanding percentile ranks via pandas (C-accelerated) ───────
    feat_cols = ["oi", "vpin", "ar", "ss"]
    feat_arrays = {c: np.array([bars[i][c] for i in range(n)], dtype=float) for c in feat_cols}
    df_tmp = pd.DataFrame(feat_arrays)

    for c in feat_cols:
        ranks = df_tmp[c].expanding().rank(pct=True) * 100
        for i in range(lookback):
            bars[i][c + "_pct"] = 0.0
        for i in range(lookback, n):
            bars[i][c + "_pct"] = ranks.iloc[i]


# ─────────────────────────────────────────────────────────────
#  Threshold Configuration (calibrated from data)
# ─────────────────────────────────────────────────────────────

# Grid-search calibrated on 30d Binance BTCUSDT 1m (2026-04-28 to 2026-05-28)
# Best config per intent: highest avg60 return over all horizons with n>=10.
THRESHOLDS = {
    # ── ABSORPTION ───────────────────────────────────────────────────────────
    # ar_pct ≥ 88 + oi_pct ≤ 8 + flat price (|ta|≤7bps), 2 consecutive bars
    # n=2199, wr60=61.4%, avg60=+4.956bps, sharpe=0.183
    "absorption": dict(
        ar_pct_min = 88,    # top 12% AR (passive buying dominant)
        oi_pct_max = 8,     # bottom 8% OI (selling being absorbed)
        ta_max_abs = 7.0,   # price flat: |ret_1m| < 7bps
        lookback   = 2,     # 2 consecutive bars
        conf_base  = 0.60,
    ),
    # ── DISTRIBUTION ─────────────────────────────────────────────────────────
    # ar_pct ≤ 10 + oi_pct ≥ 90 + flat price (|ta|≤4bps), 4 consecutive bars
    # n=1060, wr60=49.8%, avg60=+0.763bps, sharpe=0.025
    # Note: distribution is hard to trade short-term; consider as CONFIRMATION only
    "distribution": dict(
        ar_pct_max = 10,    # bottom 10% AR (aggressive selling dominant)
        oi_pct_min = 90,    # top 10% OI (buy pressure being absorbed)
        ta_max_abs = 4.0,
        lookback   = 4,
        conf_base  = 0.60,
    ),
    # ── LIQUIDITY_PROBE ──────────────────────────────────────────────────────
    # vpin_pct ≥ 92 + ss_pct ≥ 75, 3 consecutive bars
    # n=20, wr60=85.0%, avg60=+12.243bps, sharpe=0.739
    "liquidity_probe": dict(
        vpin_pct_min = 92,   # top 8% VPIN (informed trading)
        ss_pct_min   = 75,   # top 25% volume spike
        repeat_min   = 3,
        conf_base    = 0.60,
    ),
    # ── MICRO_DRIFT ───────────────────────────────────────────────────────────
    # NOT TRADEABLE with 1m kline data. OI signal in kline is too noisy.
    # Keep low-threshold version for research only.
    "micro_drift": dict(
        consecutive = 5,
        oi_pct_min = 50,
        drift_min = 1.0,
    ),
    # ── BULL_TRAP ────────────────────────────────────────────────────────────
    # price pumped >3bps + VPIN%≥80 + SS%≥75, next bar reverses within 3bps
    # n=4, wr60=75.0%, avg60=+66.781bps (but n=4, not statistically significant)
    # Safe version: looser post_rev to get more samples
    "bull_trap": dict(
        ret_min_bps  = 3.0,
        vpin_pct_min = 80,
        ss_pct_min   = 75,
        post_rev_max = 3.0,
        lookback     = 1,
        conf_base    = 0.60,
    ),
    # ── BEAR_TRAP ────────────────────────────────────────────────────────────
    "bear_trap": dict(
        ret_max_bps  = -3.0,
        vpin_pct_min = 80,
        ss_pct_min   = 75,
        post_rev_min = -3.0,
        lookback     = 1,
        conf_base    = 0.60,
    ),
    # ── GENERAL ──────────────────────────────────────────────────────────────
    "confidence_min": 0.60,
    "cooldown_bars": 5,
}


# ─────────────────────────────────────────────────────────────
#  Signal Classifier
# ─────────────────────────────────────────────────────────────

def classify_bar(bars: list, i: int, cfg: dict) -> tuple[Intent, float]:
    """
    Percentile-based classifier.
    All thresholds are percentile ranks (0-100), making them adaptive to regime.
    """
    b = bars[i]

    # ── ABSORPTION: high AR% + low OI% + flat price, 3 consecutive bars ────
    lb = cfg["absorption"]["lookback"]
    if i >= lb:
        consec = all(
            bars[i - j]["ar_pct"]  >= cfg["absorption"]["ar_pct_min"]  and
            bars[i - j]["oi_pct"]  <= cfg["absorption"]["oi_pct_max"]  and
            abs(bars[i - j]["ta"]) <= cfg["absorption"]["ta_max_abs"]
            for j in range(lb)
        )
        if consec:
            ar_ex = (b["ar_pct"]  - cfg["absorption"]["ar_pct_min"])  / (100 - cfg["absorption"]["ar_pct_min"])
            oi_ex = (cfg["absorption"]["oi_pct_max"]  - b["oi_pct"]) / cfg["absorption"]["oi_pct_max"]
            ret_sc= max(0, 1.0 - abs(b["ta"]) / cfg["absorption"]["ta_max_abs"])
            conf  = min(0.25 + ar_ex * 0.40 + oi_ex * 0.25 + ret_sc * 0.15, 1.0)
            return Intent.ABSORPTION, max(conf, cfg["absorption"]["conf_base"])

    # ── DISTRIBUTION: low AR% + high OI% + flat price, 3 consecutive bars ─
    lb = cfg["distribution"]["lookback"]
    if i >= lb:
        consec = all(
            bars[i - j]["ar_pct"]  <= cfg["distribution"]["ar_pct_max"]  and
            bars[i - j]["oi_pct"]  >= cfg["distribution"]["oi_pct_min"]  and
            abs(bars[i - j]["ta"]) <= cfg["distribution"]["ta_max_abs"]
            for j in range(lb)
        )
        if consec:
            ar_sc = (cfg["distribution"]["ar_pct_max"]  - b["ar_pct"])  / cfg["distribution"]["ar_pct_max"]
            oi_sc = (b["oi_pct"] - cfg["distribution"]["oi_pct_min"])  / (100 - cfg["distribution"]["oi_pct_min"])
            ret_sc= max(0, 1.0 - abs(b["ta"]) / cfg["distribution"]["ta_max_abs"])
            conf  = min(0.25 + ar_sc * 0.35 + oi_sc * 0.30 + ret_sc * 0.10, 1.0)
            return Intent.DISTRIBUTION, max(conf, cfg["distribution"]["conf_base"])

    # ── LIQUIDITY_PROBE: high VPIN% + high SS%, 2 consecutive bars ───────────
    if b["vpin_pct"] >= cfg["liquidity_probe"]["vpin_pct_min"] and \
       b["ss_pct"]   >= cfg["liquidity_probe"]["ss_pct_min"]:
        consec = sum(
            1 for j in range(min(i + 1, cfg["liquidity_probe"]["repeat_min"]))
            if bars[i - j]["vpin_pct"] >= cfg["liquidity_probe"]["vpin_pct_min"] and
               bars[i - j]["ss_pct"]   >= cfg["liquidity_probe"]["ss_pct_min"]
        )
        if consec >= cfg["liquidity_probe"]["repeat_min"]:
            vpin_ex = (b["vpin_pct"] - cfg["liquidity_probe"]["vpin_pct_min"]) / (100 - cfg["liquidity_probe"]["vpin_pct_min"])
            ss_ex   = (b["ss_pct"]   - cfg["liquidity_probe"]["ss_pct_min"])   / (100 - cfg["liquidity_probe"]["ss_pct_min"])
            conf = min(0.40 + vpin_ex * 0.35 + ss_ex * 0.25, 1.0)
            return Intent.LIQUIDITY_PROBE, max(conf, cfg["liquidity_probe"]["conf_base"])

    # ── MICRO_DRIFT: OI% high + micro-price drift, 3 consecutive bars ───────
    lb = cfg["micro_drift"]["consecutive"]
    if i >= lb:
        consec = all(
            bars[i - j]["oi_pct"]   >= cfg["micro_drift"]["oi_pct_min"]   and
            bars[i - j]["mp_drift"] >= cfg["micro_drift"]["drift_min"]
            for j in range(lb)
        )
        if consec:
            oi_ex = (b["oi_pct"] - cfg["micro_drift"]["oi_pct_min"]) / (100 - cfg["micro_drift"]["oi_pct_min"])
            conf  = min(0.50 + oi_ex * 0.30, 1.0)
            return Intent.MICRO_DRIFT, max(conf, cfg["micro_drift"]["conf_base"])

    # ── BULL_TRAP: price pumped >N bps, VPIN%+SS% high, next bar reverses ───
    if i + 1 < len(bars) and i >= cfg["bull_trap"]["lookback"]:
        b_next = bars[i + 1]
        if (b["ta"]                           >= cfg["bull_trap"]["ret_min_bps"]   and
            b["vpin_pct"]                     >= cfg["bull_trap"]["vpin_pct_min"]  and
            b["ss_pct"]                       >= cfg["bull_trap"]["ss_pct_min"]   and
            -cfg["bull_trap"]["post_rev_max"] <= b_next["ta"] <= cfg["bull_trap"]["post_rev_max"]):
            vpin_ex = (b["vpin_pct"] - cfg["bull_trap"]["vpin_pct_min"]) / (100 - cfg["bull_trap"]["vpin_pct_min"])
            conf = min(0.50 + vpin_ex * 0.30, 1.0)
            return Intent.BULL_TRAP, max(conf, cfg["bull_trap"]["conf_base"])

    # ── BEAR_TRAP: price dropped >N bps, VPIN%+SS% high, next bar reverses up ─
    if i + 1 < len(bars) and i >= cfg["bear_trap"]["lookback"]:
        b_next = bars[i + 1]
        if (b["ta"]                           <= cfg["bear_trap"]["ret_max_bps"]  and
            b["vpin_pct"]                     >= cfg["bear_trap"]["vpin_pct_min"]  and
            b["ss_pct"]                       >= cfg["bear_trap"]["ss_pct_min"]    and
            cfg["bear_trap"]["post_rev_min"]  <= b_next["ta"] <= -cfg["bear_trap"]["post_rev_min"]):
            vpin_ex = (b["vpin_pct"] - cfg["bear_trap"]["vpin_pct_min"]) / (100 - cfg["bear_trap"]["vpin_pct_min"])
            conf = min(0.50 + vpin_ex * 0.30, 1.0)
            return Intent.BEAR_TRAP, max(conf, cfg["bear_trap"]["conf_base"])

    return Intent.NEUTRAL, 0.0


# ─────────────────────────────────────────────────────────────
#  Forward Returns
# ─────────────────────────────────────────────────────────────

def eval_fwd(bars: list, idx: int) -> dict:
    entry_px = bars[idx]["close"]
    results = {}
    for h in [5, 15, 30, 60]:
        fidx = idx + h
        if fidx < len(bars):
            ret = (bars[fidx]["close"] - entry_px) / entry_px * 10_000
        else:
            ret = np.nan
        label = "1h" if h == 60 else f"{h}min"
        results[label] = ret
    return results


# ─────────────────────────────────────────────────────────────
#  Run Backtest
# ─────────────────────────────────────────────────────────────

def run_backtest(symbol: str = "BTCUSDT", days: int = 14, lookback: int = 20,
                 confidence_thresh: float = 0.60, args=None):
    now = int(time.time() * 1000)
    start = now - days * 86400 * 1000

    print(f"\n{'='*68}")
    print(f"  Institutional Intent Backtest  {symbol}  {days}d  lb={lookback}")
    print(f"{'='*68}\n")

    # ── load data: local parquet or live ───────────────────────────
    if args.local:
        path = f"/home/raphael/robot/data/{symbol}_1m_{days}d.parquet"
        import os
        if os.path.exists(path):
            print(f"  Loading from {path}")
            df = pd.read_parquet(path)
            bars = parse_bars_from_df(df)
            print(f"  {len(bars)} bars loaded  ({bars[0]['t']} → {bars[-1]['t']})")
        else:
            print(f"  File not found: {path}"); return
    else:
        print("  Loading 1m klines...")
        raw = fetch_klines(symbol, "1m", start, limit=1000)
        bars = parse_bars(raw)
        print(f"  {len(bars)} bars loaded  ({datetime.fromtimestamp(bars[0]['t']/1000, tz=timezone.utc).date()}"
              f" → {datetime.fromtimestamp(bars[-1]['t']/1000, tz=timezone.utc).date()})")

    print("  Computing rolling features...")
    compute_rolling(bars, lookback=lookback)

    print("  Classifying signals...")
    signals = []
    for i in range(lookback, len(bars)):  # need warmup
        intent, conf = classify_bar(bars, i, THRESHOLDS)
        if intent != Intent.NEUTRAL and conf >= confidence_thresh:
            signals.append({"intent": intent, "idx": i, "bar": bars[i], "conf": conf})

    print(f"  {len(signals)} signals above {confidence_thresh:.0%} confidence\n")

    # ── baseline ────────────────────────────────────────────────
    closes = [b["close"] for b in bars]
    bh = [(closes[i]-closes[i-1])/closes[i-1]*10_000 for i in range(1,len(closes))]
    bh_mean = np.mean(bh)
    bh_std  = np.std(bh)
    bh_wr   = sum(1 for r in bh if r>0)/len(bh)
    print(f"  Baseline buy-hold:  win={bh_wr:.1%}  avg={bh_mean:+.2f}bps  std={bh_std:.1f}bps")

    # ── group forward returns by intent ──────────────────────────
    by_intent: dict = defaultdict(lambda: {"5min":[],"15min":[],"30min":[],"1h":[]})
    for sig in signals:
        fwd = eval_fwd(bars, sig["idx"])
        for label, ret in fwd.items():
            if not np.isnan(ret):
                by_intent[sig["intent"].value][label].append(ret)

    # ── print table ─────────────────────────────────────────────
    print(f"\n  Signal Performance vs BH baseline ({bh_mean:+.2f}bps avg):")
    print(f"  {'Intent':<20} {'Horizon':>8} {'N':>4} {'WinRate':>8} {'AvgBps':>9} {'MedBps':>9} {'Sharpe':>7} {'VsBH':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*4} {'-'*8} {'-'*9} {'-'*9} {'-'*7} {'-'*8}")

    intent_order = ["absorption","distribution","liquidity_probe","micro_drift","bull_trap","bear_trap"]
    for intent_str in intent_order:
        if intent_str not in by_intent:
            continue
        for h, label in [(5,"5min"),(15,"15min"),(30,"30min"),(60,"1h")]:
            pnls = by_intent[intent_str][label]
            if len(pnls) < 4:
                continue
            wr   = sum(1 for p in pnls if p>0)/len(pnls)
            avg  = np.mean(pnls)
            med  = np.median(pnls)
            std  = np.std(pnls) or 1.0
            shp  = (avg - bh_mean) / std
            vs   = avg - bh_mean
            mark = "✓" if shp > 0.3 and avg > 5 else "✗" if shp < -0.3 and avg < -5 else " "
            print(f"  {mark} {intent_str:<20} {h:>4}m {len(pnls):>4} {wr:>7.1%} {avg:>+8.1f} {med:>+8.1f} {shp:>7.2f} {vs:>+7.1f}")

    # ── signal counts ────────────────────────────────────────────
    from collections import Counter
    cnt = Counter(s["intent"].value for s in signals)
    print(f"\n  Signal counts:")
    for k, v in cnt.most_common():
        print(f"    {k}: {v}")

    # ── recent signals detail ────────────────────────────────────
    print(f"\n  Last 10 signals:")
    print(f"  {'Time(UTC)':<14} {'Intent':<20} {'Conf':>5} {'Price':>10} {'Ret1m':>7} {'OI':>7} {'VPIN':>6} {'AR':>6} {'SS':>5} {'Drift':>7}")
    print(f"  {'-'*14} {'-'*20} {'-'*5} {'-'*10} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*7}")
    for sig in signals[-10:]:
        b = sig["bar"]
        ts = datetime.fromtimestamp(b["t"]/1000, tz=timezone.utc).strftime("%H:%M %m/%d")
        print(f"  {ts}  {sig['intent'].value:<20} {sig['conf']:.0%} {b['close']:>10.4f} "
              f"{b['ret_1m']:>+7.1f} {b['oi']:>+7.2f} {b['vpin']:>6.3f} {b['ar']:>6.2f} "
              f"{b['ss']:>5.2f} {b['mp_drift']:>6.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.60)
    parser.add_argument("--local", action="store_true", help="Load from local parquet instead of fetching live")
    args = parser.parse_args()
    run_backtest(args.symbol, args.days, args.lookback, args.conf, args)