"""
live_analysis.py
================
Fetch real-time data from Binance, compute all order-flow metrics,
and classify institutional intent signals live.

Usage:
    python live_analysis.py [--symbol BTCUSDT] [--duration 60]
"""

import argparse
import time
import json
import requests
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np

# ── small local imports ──────────────────────────────────────
import sys
sys.path.insert(0, "/home/raphael/robot/src")
from core.orderflow import (
    Intent, IntentSignal, InstitutionalIntentClassifier,
    MicroPriceEngine, OrderBookLevel, OrderBookSnap,
    OrderFlowMetrics, Trade, format_signal,
)


# ─────────────────────────────────────────────────────────────
#  Binance REST fetcher (no WS needed — REST polling is fine
#  for live analysis at 1-5 second intervals)
# ─────────────────────────────────────────────────────────────

class BinanceFetcher:
    BASE = "https://api.binance.com/api/v3"

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBookSnap:
        r = requests.get(f"{self.BASE}/depth", params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return OrderBookSnap(
            exchange="binance",
            symbol=symbol,
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=float(p), size=float(q)) for p, q in d["bids"]],
            asks=[OrderBookLevel(price=float(p), size=float(q)) for p, q in d["asks"]],
        )

    def fetch_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        r = requests.get(f"{self.BASE}/aggTrades", params={"symbol": symbol, "limit": limit}, timeout=10)
        r.raise_for_status()
        trades = r.json()
        return [
            Trade(
                exchange="binance",
                symbol=symbol,
                ts_ms=t["T"],
                price=float(t["p"]),
                size=float(t["q"]),
                side="buy" if not t["m"] else "sell",   # m=False → buyer taker → aggressive buy
                is_aggressive=not t["m"],
            )
            for t in trades
        ]

    def fetch_klines(self, symbol: str, interval: str = "1m", limit: int = 5):
        r = requests.get(f"{self.BASE}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────
#  Live Analysis Runner
# ─────────────────────────────────────────────────────────────

@dataclass
class LiveAnalysis:
    symbol: str = "BTCUSDT"
    poll_interval_sec: float = 2.0
    duration_sec: float = 60.0
    verbose: bool = True

    _clf: Optional[InstitutionalIntentClassifier] = field(default=None, init=False)
    _mp: Optional[MicroPriceEngine] = field(default=None, init=False)
    _flow: Optional[OrderFlowMetrics] = field(default=None, init=False)
    _fetcher: Optional[BinanceFetcher] = field(default=None, init=False)
    _history: list[OrderBookSnap] = field(default_factory=list, init=False)
    _signals: list[IntentSignal] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._clf = InstitutionalIntentClassifier()
        self._mp = MicroPriceEngine()
        self._flow = OrderFlowMetrics()
        self._fetcher = BinanceFetcher()

    def run(self):
        end_time = time.time() + self.duration_sec
        iteration = 0

        print(f"\n{'='*60}")
        print(f"  LIVE ANALYSIS  {self.symbol}  ·  poll={self.poll_interval_sec}s  ·  {self.duration_sec}s")
        print(f"{'='*60}\n")

        while time.time() < end_time:
            iteration += 1
            try:
                self._tick(iteration)
            except Exception as e:
                print(f"  [ERROR] {e}")

            time.sleep(self.poll_interval_sec)

        self._summary()

    def _tick(self, iteration: int):
        # ── fetch data ─────────────────────────────────────────
        snap = self._fetcher.fetch_orderbook(self.symbol)
        trades = self._fetcher.fetch_recent_trades(self.symbol, limit=100)
        klines = self._fetcher.fetch_klines(self.symbol, limit=3)

        self._history.append(snap)
        if len(self._history) > 200:
            self._history = self._history[-200:]

        # ── update engines ─────────────────────────────────────
        self._mp.ingest_orderbook(snap)
        for t in trades:
            self._flow.ingest_trade(t)
            self._mp.ingest_trade(t)

        self._clf.update_orderbook(snap)
        for t in trades:
            self._clf.update_trade(t)

        # ── compute metrics ────────────────────────────────────
        metrics = self._flow.ingest_orderbook(snap)
        mp_state = self._mp.ingest_orderbook(snap)
        oi = metrics["order_imbalance"]
        vpin = metrics["vpin"]
        ar = metrics["absorption_ratio"]
        ss = metrics["spoofing_score"]
        ta = metrics["trade_aggression"]

        spread_bps = snap.spread_bps
        drift_bps = abs(mp_state.micro_price - mp_state.mid_price) / mp_state.mid_price * 10_000

        # ── classify ────────────────────────────────────────────
        sig = self._clf.classify(snap, trades)
        if sig.intent != Intent.NEUTRAL and sig.confidence >= 0.6:
            self._signals.append(sig)
            flag = f"*** {format_signal(sig)} ***"
        else:
            flag = ""

        # ── print row ──────────────────────────────────────────
        now = datetime.now().strftime("%H:%M:%S")
        kclose = float(klines[-1][4]) if klines else snap.mid_price
        direction = "▲" if mp_state.micro_price > mp_state.mid_price else "▼"
        imbalance_bar = "▓" * max(0, int(abs(oi) * 10)) + "░" * max(0, int((1 - abs(oi)) * 10))

        print(
            f"  {now}  iter={iteration:3d}  "
            f"mid={snap.mid_price:>10.2f}  "
            f"klose={kclose:>10.2f}  "
            f"spread={spread_bps:>5.1f}bps  "
            f"oi={oi:>+5.2f} {imbalance_bar}  "
            f"vpin={vpin:.3f}  ar={ar:.2f}  ss={ss:.2f}  "
            f"micro={mp_state.micro_price:>10.4f}{direction}  drift={drift_bps:>4.1f}bps  "
            f"{flag}"
        )

    def _summary(self):
        print(f"\n{'='*60}")
        print(f"  SUMMARY  ({len(self._history)} snapshots)")
        print(f"{'='*60}")

        if not self._signals:
            print("  No signals fired.")
            return

        counts: dict[str, int] = {}
        for s in self._signals:
            counts[s.intent.value] = counts.get(s.intent.value, 0) + 1

        print(f"\n  Signal counts:")
        for intent, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {intent}: {count}")

        print(f"\n  Last 5 signals:")
        for s in self._signals[-5:]:
            ts = datetime.fromtimestamp(s.ts_ms / 1000).strftime("%H:%M:%S")
            print(f"    {ts}  {s.intent.value}  conf={s.confidence:.0%}  price={s.price_at_signal:.2f}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live institutional intent analysis")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    LiveAnalysis(symbol=args.symbol, poll_interval_sec=args.poll, duration_sec=args.duration).run()