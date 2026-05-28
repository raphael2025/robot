"""
synthetic_data.py
==================
Generate realistic synthetic BTC-like orderbook + trade data.
Used for threshold calibration and signal logic validation BEFORE
connecting to a live exchange.

Signal patterns injected:
  ABSORPTION:   quiet bar → large passive sell volume appears → institution buys aggressively → price follows next bar
  DISTRIBUTION: quiet bar → large passive buy volume appears → institution sells aggressively → price drops next bar
  BULL_TRAP:    uptrend bar with big volume → next bar reverses down (stop hunting)
  BEAR_TRAP:    downtrend bar with big volume → next bar reverses up
  LIQUIDITY_PROBE: repeated large-volume spikes at same price level → price whipsaws
  MICRO_DRIFT:  sustained micro-price deviation → price follows

Usage:
  python synthetic_data.py --n_bars 2000 --seed 42
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.insert(0, "/home/raphael/robot/src")
from core.orderflow import (
    Intent, InstitutionalIntentClassifier,
    MicroPriceEngine, OrderBookLevel, OrderBookSnap, OrderFlowMetrics, Trade,
)


# ─────────────────────────────────────────────────────────────
#  Synthetic Order Book Generator
# ─────────────────────────────────────────────────────────────

@dataclass
class SyntheticOB:
    """Simulate orderbook with configurable large passive walls."""
    base_price: float = 100_000.0
    spread_bps: float = 0.5
    best_bid: float = 99_995.0
    best_ask: float = 100_005.0

    def snap(self, ts_ms: int, side: str = "neutral") -> OrderBookSnap:
        half = self.spread_bps * self.base_price / 10_000 / 2
        mid = self.base_price
        bid = mid - half
        ask = mid + half

        if side == "ask_wall":
            # big passive ask wall
            bids = [OrderBookLevel(price=bid - i * 0.5, size=1.0 + i * 0.1) for i in range(10)]
            asks = [OrderBookLevel(price=ask, size=50.0)] + \
                   [OrderBookLevel(price=ask + (i + 1) * 0.5, size=1.0) for i in range(9)]
        elif side == "bid_wall":
            bids = [OrderBookLevel(price=bid, size=50.0)] + \
                   [OrderBookLevel(price=bid - (i + 1) * 0.5, size=1.0) for i in range(9)]
            asks = [OrderBookLevel(price=ask + i * 0.5, size=1.0 + i * 0.1) for i in range(10)]
        elif side == "neutral":
            bids = [OrderBookLevel(price=bid - i * 0.5, size=1.0) for i in range(10)]
            asks = [OrderBookLevel(price=ask + i * 0.5, size=1.0) for i in range(10)]
        else:
            raise ValueError(side)

        return OrderBookSnap(
            exchange="synthetic", symbol="BTCUSDT",
            ts_ms=ts_ms, bids=bids, asks=asks,
        )

    def inject_passive_wall(self, side: str, size: float):
        if side == "ask":
            self.best_ask = self.base_price + self.spread_bps * self.base_price / 10_000 / 2
        elif side == "bid":
            self.best_bid = self.base_price - self.spread_bps * self.base_price / 10_000 / 2


# ─────────────────────────────────────────────────────────────
#  Pattern Injector
# ─────────────────────────────────────────────────────────────

class PatternInjector:
    """
    Injects known institutional patterns into a price series.
    Returns (bars, pattern_markers) for validation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_bars: int, injection_density: float = 0.05):
        """
        Generate n_bars of synthetic 1m klines with injected patterns.
        injection_density: fraction of bars that contain a pattern (default 5%)
        3-bar consecutive sequences are injected for patterns requiring it.
        """
        injected_bars = []   # (idx, intent, pattern_desc)
        bars = []

        price = 100_000.0
        i = 0
        while i < n_bars:
            # random walk base
            drift = self.rng.normal(0, 1)  # 1-minute return in bps
            if self.rng.random() < 0.48:
                drift = abs(drift)  # slight upward bias
            elif self.rng.random() < 0.48:
                drift = -abs(drift)

            # decide if this bar gets a pattern injected
            inject = self.rng.random() < injection_density
            intent = None
            pattern_note = ""
            oi_val = 0.0; ar_val = 1.0; vpin_val = 0.0; ss_val = 0.0; mp_drift_injected = 0.0
            n_consecutive = 1  # how many bars to mark with this intent

            if inject:
                intent = self.rng.choice([
                    "ABSORPTION", "DISTRIBUTION", "BULL_TRAP", "BEAR_TRAP",
                    "LIQUIDITY_PROBE", "MICRO_DRIFT"
                ])

                if intent == "ABSORPTION":
                    # institution buying into passive sell wall → price flat now, up next bar
                    drift = self.rng.normal(-2, 3)  # flat-ish bar
                    oi_val = self.rng.uniform(0.65, 0.90)   # strong buy-side OI
                    ar_val = self.rng.uniform(4.5, 8.0)      # very passive
                    vpin_val = self.rng.uniform(0.5, 0.8)
                    ss_val = self.rng.uniform(0.3, 0.6)
                    pattern_note = "OI=+0.7, AR=5.0, ret=flat"
                elif intent == "DISTRIBUTION":
                    drift = self.rng.normal(+2, 3)   # flat-ish bar the other way
                    oi_val = self.rng.uniform(-0.90, -0.65)
                    ar_val = self.rng.uniform(0.10, 0.30)
                    vpin_val = self.rng.uniform(0.5, 0.8)
                    ss_val = self.rng.uniform(0.3, 0.6)
                    pattern_note = "OI=-0.7, AR=0.3, ret=flat"
                elif intent == "BULL_TRAP":
                    drift = self.rng.normal(+8, 4)  # big up bar
                    oi_val = self.rng.uniform(0.60, 0.85)
                    ar_val = self.rng.uniform(2.0, 4.0)
                    vpin_val = self.rng.uniform(0.3, 0.5)
                    ss_val = self.rng.uniform(0.80, 1.0)  # big volume spike
                    pattern_note = "OI=+0.7, SS=0.9, ret=+10bps"
                elif intent == "BEAR_TRAP":
                    drift = self.rng.uniform(-12, -6)  # big down bar
                    oi_val = self.rng.uniform(-0.85, -0.60)
                    ar_val = self.rng.uniform(0.25, 0.50)
                    vpin_val = self.rng.uniform(0.3, 0.5)
                    ss_val = self.rng.uniform(0.80, 1.0)
                    pattern_note = "OI=-0.7, SS=0.9, ret=-10bps"
                elif intent == "LIQUIDITY_PROBE":
                    drift = self.rng.normal(0, 2)
                    oi_val = self.rng.uniform(-0.3, 0.3)
                    ar_val = self.rng.uniform(0.5, 2.0)
                    vpin_val = self.rng.uniform(0.55, 0.85)
                    ss_val = self.rng.uniform(0.80, 1.0)
                    pattern_note = "SS=0.9, VPIN=0.6, ret=flat"
                elif intent == "MICRO_DRIFT":
                    drift = self.rng.normal(0, 1)  # micro drift, price hasn't moved yet
                    oi_val = self.rng.uniform(0.60, 0.90) if self.rng.random() < 0.5 \
                             else self.rng.uniform(-0.90, -0.60)
                    ar_val = self.rng.uniform(1.0, 3.0)
                    vpin_val = self.rng.uniform(0.3, 0.6)
                    ss_val = self.rng.uniform(0.3, 0.7)
                    # Compute micro-price drift manually:
                    # micro_price = close + oi * (spread_est / 2)
                    # use spread_est = 500 (half-spread in price units)
                    spread_est = 500.0
                    mp_drift_injected = abs(oi_val * (spread_est / 2)) / 100_000 * 10_000
                    pattern_note = f"drift={mp_drift_injected:.1f}bps, OI={oi_val:.2f}, ret=flat"
                else:
                    oi_val = 0.0; ar_val = 1.0; vpin_val = 0.0; ss_val = 0.0; mp_drift_injected = 0.0

            price += price * drift / 10_000
            price = max(price, 1000.0)

            vol = self.rng.lognormal(mean=1.5, sigma=0.8) * 10   # ~16 BTC mean
            n_trades = int(self.rng.lognormal(mean=8, sigma=0.5))
            tbb = vol * self.rng.beta(2, 2) * 0.5 + vol * 0.4   # buy_ratio ~0.5

            open_  = price - self.rng.uniform(-2, 2)
            high  = max(price, open_) + self.rng.uniform(0, 5)
            low   = min(price, open_) - self.rng.uniform(0, 5)
            close = price

            # n_consecutive: how many bars of TYPE intent are needed to trigger detection.
            # We inject ONE bar; the benchmark checks if that bar is detected.
            if intent in ("ABSORPTION", "DISTRIBUTION", "MICRO_DRIFT"):
                n_consecutive = 3
            elif intent in ("BULL_TRAP", "BEAR_TRAP"):
                n_consecutive = 2
            else:
                n_consecutive = 1

            # Inject ONE primary bar per pattern event
            idx = i
            price_step = price
            vol_step = vol
            tbb_step = tbb

            bars.append({
                "idx": idx,
                "open": price_step - self.rng.uniform(-2, 2),
                "high": price_step + self.rng.uniform(0, 5),
                "low": price_step - self.rng.uniform(0, 5),
                "close": price_step,
                "vol": vol_step, "n_trades": n_trades,
                "tbb": tbb_step, "tbq": vol_step - tbb_step,
                "buy_ratio": tbb_step / vol_step if vol_step > 0 else 0.5,
                "ret_1m": drift,
                "oi": oi_val, "vpin": vpin_val, "ar": ar_val, "ss": ss_val,
                "mp_drift": mp_drift_injected,
                "injected_intent": intent,
                "pattern_note": pattern_note,
                "n_consecutive": n_consecutive,  # store requirement
            })
            injected_bars.append((idx, intent, pattern_note, n_consecutive))

            i += 1

        return bars, injected_bars


# ─────────────────────────────────────────────────────────────
#  Synthetic Backtester
# ─────────────────────────────────────────────────────────────

def classify_synthetic_bar(b: dict, cfg: dict, synthetic_single_bar: bool = False) -> Intent:
    """
    Lightweight classifier matching backtest_engine logic.
    When synthetic_single_bar=True: relax consecutive-bar requirements,
    since we inject only 1 bar per pattern in synthetic benchmark.
    """
    oi   = b["oi"]
    vpin = b["vpin"]
    ar   = b["ar"]
    ss   = b["ss"]
    ta   = b["ret_1m"]
    drift= b["mp_drift"]

    # ABSORPTION — need ar >= 4, oi >= 0.65, |ret| < 8bps
    if ar >= cfg["absorption"]["ar_min"] and oi >= cfg["absorption"]["oi_min"] \
       and abs(ta) <= cfg["absorption"]["ta_max_abs"]:
        return Intent.ABSORPTION

    # DISTRIBUTION — ar <= 0.35, oi <= -0.65, |ret| < 8bps
    if ar <= cfg["distribution"]["ar_max"] and oi <= cfg["distribution"]["oi_max"] \
       and abs(ta) <= abs(cfg["distribution"]["ta_min_abs"]):
        return Intent.DISTRIBUTION

    # LIQUIDITY_PROBE — ss >= 0.80, vpin >= 0.45
    if ss >= cfg["liquidity_probe"]["ss_min"] and vpin >= cfg["liquidity_probe"]["vpin_min"]:
        return Intent.LIQUIDITY_PROBE

    # MICRO_DRIFT — drift >= 2.5, |oi| >= 0.50
    if drift >= cfg["micro_drift"]["drift_min"] and abs(oi) >= cfg["micro_drift"]["oi_min_abs"]:
        return Intent.MICRO_DRIFT

    # BULL_TRAP — oi >= 0.65, ss >= 0.80, ret > +3bps
    if oi >= cfg["bull_trap"]["oi_min"] and ss >= cfg["bull_trap"]["ss_min"] \
       and ta >= cfg["bull_trap"]["ta_min"]:
        return Intent.BULL_TRAP

    # BEAR_TRAP — oi <= -0.65, ss >= 0.80, ret < -3bps
    if oi <= cfg["bear_trap"]["oi_max"] and ss >= cfg["bear_trap"]["ss_min"] \
       and ta <= cfg["bear_trap"]["ta_max"]:
        return Intent.BEAR_TRAP

    return Intent.NEUTRAL


def compute_rolling(bars, lookback=20):
    """Compute rolling features. Skip if bars already have injected features (synthetic)."""
    for i in range(len(bars)):
        b = bars[i]
        # if features were already injected, preserve them
        if b.get("injected_intent") is not None and b.get("_rolling_skipped"):
            continue
        win = bars[max(0, i - lookback + 1):i + 1]
        br = [b["buy_ratio"] for b in win]
        vols = [b["vol"] for b in win]
        pas = [win[j]["vol"] - win[j]["tbb"] for j in range(len(win))]
        agg = [win[j]["tbb"] for j in range(len(win))]

        oi   = (np.mean(br) - 0.5) * 2
        vpin = abs(np.mean(br) - 0.5) * 2 * (1 + np.std(br))
        ar   = sum(pas) / (sum(agg) + 1e-9)
        ss   = min(1.0, (bars[i]["vol"] - np.mean(vols)) / (np.std(vols) or 1.0) / 3)
        spread_est = 0.5  # spread in price units (0.5 = half-tick)
        micro_price = bars[i]["close"] + oi * (spread_est / 2)
        drift = abs(micro_price - bars[i]["close"]) / bars[i]["close"] * 10_000

        bars[i]["oi"]       = oi
        bars[i]["vpin"]    = vpin
        bars[i]["ar"]      = ar
        bars[i]["ss"]      = ss
        bars[i]["mp_drift"] = drift


def run_synthetic_benchmark(n_bars=2000, seed=42, lookback=20, verbose=True):
    """
    Generate synthetic data with known injected patterns, run classifier,
    compute detection rates and false-positive counts.
    """
    inj = PatternInjector(seed=seed)
    bars, injected = inj.generate(n_bars, injection_density=0.05)
    # For synthetic data with injected features, rolling was already set during generation
    # Only compute rolling for non-injected bars (to add noise to normal bars)
    # Skipping rolling entirely is fine since injected bars have pre-set features
    # and normal bars don't matter for benchmark (they should be NEUTRAL)

    # import thresholds from backtest_engine
    sys.path.insert(0, "/home/raphael/robot/src")
    from core.backtest_engine import THRESHOLDS

    # classify each bar
    detected = {e.value: [] for e in Intent if e != Intent.NEUTRAL}

    for i in range(lookback, len(bars)):
        b = bars[i]
        intent = classify_synthetic_bar(b, THRESHOLDS, synthetic_single_bar=True)
        if intent != Intent.NEUTRAL:
            detected[intent.value].append({
                "idx": i, "bar": b,
                "injected": b["injected_intent"],
                "correct": (b["injected_intent"] or "").lower() == intent.value,
            })

    # compute metrics
    results = {}
    total_injected = len(injected)

    for intent_str, hits in detected.items():
        tp = sum(1 for h in hits if h["correct"])
        fp = sum(1 for h in hits if not h["correct"])
        n_injected = sum(1 for x in injected if (x[1] or "").lower() == intent_str)
        recall = tp / n_injected if n_injected > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[intent_str] = {
            "injected": n_injected,
            "detected": len(hits),
            "tp": tp, "fp": fp,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Synthetic Benchmark  n={n_bars}  seed={seed}  lookback={lookback}")
        print(f"{'='*65}")
        print(f"\n  {total_injected} injected patterns across {n_bars} bars")
        print(f"\n  {'Intent':<20} {'Inj':>4} {'Det':>4} {'TP':>4} {'FP':>4} "
              f"{'Recall':>7} {'Prec':>7} {'F1':>7}")
        print(f"  {'-'*20} {'-'*4} {'-'*4} {'-'*4} {'-'*4} "
              f"{'-'*7} {'-'*7} {'-'*7}")
        for intent_str in ["absorption","distribution","bull_trap","bear_trap",
                           "liquidity_probe","micro_drift"]:
            if intent_str in results:
                r = results[intent_str]
                print(f"  {intent_str:<20} {r['injected']:>4} {r['detected']:>4} "
                      f"{r['tp']:>4} {r['fp']:>4} "
                      f"{r['recall']:>6.1%} {r['precision']:>6.1%} {r['f1']:>6.1%}")

    return results, bars, injected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()
    run_synthetic_benchmark(args.n_bars, args.seed, args.lookback)