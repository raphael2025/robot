"""
Institution Intent Detection Engine
===================================
Classifies institutional order flow into 6 patterns:
  1. 被动买单堆积 + 突然撤单  → 诱多 (Bull Trap)
  2. 被动卖单堆积 + 突然撤单  → 诱空 (Bear Trap)
  3. 主动买入 + 价格不跟涨    → 收集 (Absorption)
  4. 主动卖出 + 价格不跟跌    → 派发 (Distribution)
  5. 被动挂单 + 反复撤挂      → 流动性测试/诱导 (Liquidity Probe)
  6. Micro-Price 持续偏离     → 方向信号 (Micro-Price Drift)

Architecture:
  ExchangeAdapter (WebSocket/REST) → OrderBookSnap → MicroPriceEngine
    → InstitutionalIntentClassifier → IntentSignal → [Alert / TradingEngine]

Author: Raphael Quant Team
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("robot")


# ─────────────────────────────────────────────
#  Enums & Dataclasses
# ─────────────────────────────────────────────

class Intent(Enum):
    BULL_TRAP   = "bull_trap"    # 被动买单堆积 + 突然撤单 → 准备砸盘
    BEAR_TRAP   = "bear_trap"    # 被动卖单堆积 + 突然撤单 → 准备拉升
    ABSORPTION  = "absorption"    # 主动买入 + 价格不跟涨 → 暗中吸货
    DISTRIBUTION= "distribution" # 主动卖出 + 价格不跟跌 → 暗中派发
    LIQUIDITY_PROBE = "liquidity_probe"  # 大单挂而不成交 + 反复撤挂
    MICRO_DRIFT = "micro_drift"  # Micro-Price 持续偏离价格
    NEUTRAL     = "neutral"


class SignalStrength(Enum):
    WEAK   = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class OrderBookLevel:
    price: float
    size: float
    order_count: int = 0   # number of individual orders at this level


@dataclass
class OrderBookSnap:
    exchange: str
    symbol: str
    ts_ms: int
    bids: list[OrderBookLevel]   # best bid first, ascending
    asks: list[OrderBookLevel]   # best ask first, ascending

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def spread_bps(self) -> float:
        return self.spread / self.mid_price * 10_000 if self.mid_price else 0.0


@dataclass
class Trade:
    exchange: str
    symbol: str
    ts_ms: int
    price: float
    size: float          # quantity
    side: str            # "buy" or "sell"
    is_aggressive: bool  #主动成交 True, 被动成交 False


@dataclass
class IntentSignal:
    intent: Intent
    strength: SignalStrength
    confidence: float     # 0..1
    ts_ms: int
    symbol: str
    details: dict        # raw metrics that triggered this signal
    price_at_signal: float


@dataclass
class MicroPriceState:
    """Micro-Price = best_bid + imbalance_ratio * spread"""
    micro_price: float
    mid_price: float
    imbalance: float     # -1..1  (bid-side heavy = positive)
    bid_depth_heavy: bool
    timestamp_ms: int


# ─────────────────────────────────────────────
#  Core: Micro-Price Engine
# ─────────────────────────────────────────────

class MicroPriceEngine:
    """
    Micro-Price (Jerrett & Keene, 2019)
    ====================================
    micro_price = best_bid + imbalance * spread
    imbalance  ∈ [-1, 1]
      +1 → all volume on bid side  → price should move UP
      -1 → all volume on ask side  → price should move DOWN

    Formula:
        micro_price = mid_price + imbalance * (spread / 2)
        imbalance  = (V_bid - V_ask) / (V_bid + V_ask)

    Where V_bid / V_ask = volume weighted by time in recent window.
    """

    def __init__(self, window_ms: int = 5_000, depth_levels: int = 10):
        self.window_ms = window_ms
        self.depth_levels = depth_levels
        self._order_flow: list[tuple[int, float, float, bool]] = []  # (ts_ms, size, price, is_buy)
        self._last_imbalance = 0.0
        self._drift_count = 0
        self._drift_threshold = 3   # consecutive prints before MICRO_DRIFT fires

    def ingest_trade(self, trade: Trade) -> None:
        self._order_flow.append((trade.ts_ms, trade.size, trade.price, trade.side == "buy"))
        self._prune_old(time.time() * 1000 - self.window_ms)

    def ingest_orderbook(self, snap: OrderBookSnap) -> MicroPriceState:
        """Compute Micro-Price from current order book depth."""
        now = snap.ts_ms
        self._prune_old(now - self.window_ms)

        # compute volume imbalance from order book depth
        bid_vol = sum(l.size for l in snap.bids[: self.depth_levels])
        ask_vol = sum(l.size for l in snap.asks[: self.depth_levels])

        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

        spread = snap.spread
        mid = snap.mid_price

        # micro_price = mid + imbalance * (spread / 2)
        micro_price = mid + imbalance * (spread / 2)

        state = MicroPriceState(
            micro_price=micro_price,
            mid_price=mid,
            imbalance=imbalance,
            bid_depth_heavy=imbalance > 0,
            timestamp_ms=now,
        )

        # track consecutive drift direction
        if abs(imbalance) > 0.6:  # strong side
            drift = micro_price - mid
            if abs(drift) > spread * 0.3:
                self._drift_count += 1
            else:
                self._drift_count = 0
        else:
            self._drift_count = 0

        self._last_imbalance = imbalance
        return state

    def _prune_old(self, cutoff_ms: int) -> None:
        self._order_flow = [(t, s, p, b) for t, s, p, b in self._order_flow if t >= cutoff_ms]

    @property
    def is_micro_drift(self) -> bool:
        return self._drift_count >= self._drift_threshold


# ─────────────────────────────────────────────
#  Core: Order Flow Metrics
# ─────────────────────────────────────────────

class OrderFlowMetrics:
    """
    Computes order-flow analytics from a sliding window of trades + orderbook.

    Key metrics:
      - VPIN (Volume-synchronized Probability of Informed Trading)
      - Order Imbalance (OI)
      - Trade Aggression
      - Absorption Ratio
      - Spoofing Score
    """

    def __init__(
        self,
        window_trades: int = 100,
        vpin_buckets: int = 50,
        absorption_window: int = 50,
    ):
        self.window_trades = window_trades
        self.vpin_buckets = vpin_buckets
        self.absorption_window = absorption_window
        self._trades: list[Trade] = []
        self._vpin_bucket_size: float = 0.0
        self._vbucket: list[bool] = []  # is buy volume?

    def ingest_trade(self, trade: Trade) -> None:
        self._trades.append(trade)
        if len(self._trades) > self.window_trades * 2:
            self._trades = self._trades[-self.window_trades * 2:]

    def ingest_orderbook(self, snap: OrderBookSnap) -> dict:
        """Compute all metrics. Call after each orderbook update."""
        self._compute_vpin()
        oi = self._order_imbalance(snap)
        ta = self._trade_aggression()
        ar = self._absorption_ratio()
        ss = self._spoofing_score()
        return {
            "vpin": self._vpin,
            "order_imbalance": oi,
            "trade_aggression": ta,
            "absorption_ratio": ar,
            "spoofing_score": ss,
            "bid_depth_heavy": oi > 0,
            "trade_count": len(self._trades),
        }

    # ── VPIN ──────────────────────────────────────────────────────────────

    @property
    def _vpin(self) -> float:
        if len(self._vbucket) < self.vpin_buckets:
            return 0.0
        buy_buckets = sum(1 for b in self._vbucket if b)
        return abs(buy_buckets / self.vpin_buckets - 0.5) * 2  # 0..1

    def _compute_vpin(self) -> None:
        if len(self._trades) < self.vpin_buckets:
            return

        if self._vpin_bucket_size == 0:
            avg_trade = np.mean([t.size for t in self._trades])
            self._vpin_bucket_size = avg_trade * 10  # calibration constant

        vol = 0.0
        for t in self._trades[-self.vpin_buckets:]:
            vol += t.size
            if vol >= self._vpin_bucket_size:
                self._vbucket.append(t.side == "buy")
                vol = 0.0

        if len(self._vbucket) > self.vpin_buckets * 2:
            self._vbucket = self._vbucket[-self.vpin_buckets:]

    # ── Order Imbalance ─────────────────────────────────────────────────────

    @staticmethod
    def _order_imbalance(snap: OrderBookSnap, levels: int = 10) -> float:
        bid_vol = sum(l.size for l in snap.bids[:levels])
        ask_vol = sum(l.size for l in snap.asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total  # -1..1

    # ── Trade Aggression ────────────────────────────────────────────────────

    def _trade_aggression(self) -> float:
        """
        ΔP / Δt normalized.
        + = aggressive buys driving price up
        - = aggressive sells driving price down
        """
        if len(self._trades) < 2:
            return 0.0

        recent = self._trades[-50:]
        prices = [t.price for t in recent]
        ts = [t.ts_ms for t in recent]

        dP = prices[-1] - prices[0]
        dT = (ts[-1] - ts[0]) / 1000
        if dT == 0:
            return 0.0
        return dP / dT  # price per second

    # ── Absorption Ratio ────────────────────────────────────────────────────

    def _absorption_ratio(self) -> float:
        """
        How much of the aggressive volume was absorbed by passive orders.
        absorption = passive_volume / aggressive_volume
        > 1 = heavy passive absorption (institution accumulating)
        < 0.5 = aggressive hits through liquidity (institution distributing)
        """
        if len(self._trades) < self.absorption_window:
            return 1.0

        window = self._trades[-self.absorption_window:]
        agg_vol = sum(t.size for t in window if t.is_aggressive)
        pas_vol = sum(t.size for t in window if not t.is_aggressive)

        if agg_vol == 0:
            return 1.0
        return pas_vol / agg_vol

    # ── Spoofing Score ──────────────────────────────────────────────────────

    def _spoofing_score(self) -> float:
        """
        Heuristic: large orders appearing and disappearing without execution.
        Returns 0..1 score of spoofing likelihood.
        """
        # simple proxy: count large passive orders vs executed
        if len(self._trades) < 20:
            return 0.0

        window = self._trades[-100:]
        avg_size = np.mean([t.size for t in window])
        large_orders = sum(1 for t in window if t.size > avg_size * 5 and not t.is_aggressive)
        total_large = sum(1 for t in window if t.size > avg_size * 5)

        if total_large == 0:
            return 0.0
        return large_orders / total_large  # high score = likely spoofing


# ─────────────────────────────────────────────
#  Core: Institutional Intent Classifier
# ─────────────────────────────────────────────

@dataclass
class IntentThresholds:
    # absorption
    abs_absorption_min: float = 2.5
    abs_price_lag_max_bps: float = 5.0      # price doesn't follow by more than 5bps
    # distribution
    dist_absorption_max: float = 0.5
    # traps
    trap_imbalance_min: float = 0.70        # passive orders this side
    trap_cancel_ratio_min: float = 0.60      # % of passive volume cancelled
    # liquidity probe
    probe_passive_vol_min: float = 5.0      # × avg trade size
    probe_repeat_min: int = 3               # repeat cycles
    # micro drift
    drift_imbalance_min: float = 0.60
    drift_consecutive_min: int = 3
    # confidence weights
    vpin_strong_thresh: float = 0.6
    vpin_mod_thresh: float = 0.4


class InstitutionalIntentClassifier:
    """
    Maps order-flow metrics → Intent signals.

    Signal rules (优先级顺序):
      1. MICRO_DRIFT  — Micro-Price偏离 mid > spread×0.3 且连续3次
      2. ABSORPTION   — 主动买入↑ + 价格不跟涨 + 高absorption
      3. DISTRIBUTION — 主动卖出↑ + 价格不跟跌 + 低absorption
      4. BULL_TRAP    — 被动卖单堆积 + 突然撤单
      5. BEAR_TRAP    — 被动买单堆积 + 突然撤单
      6. LIQUIDITY_PROBE — 大单挂而不成交 + 反复撤挂
      7. NEUTRAL
    """

    def __init__(self, thresholds: Optional[IntentThresholds] = None):
        self.t = thresholds or IntentThresholds()
        self._micro = MicroPriceEngine()
        self._flow = OrderFlowMetrics()
        self._last_snap: Optional[OrderBookSnap] = None
        self._snap_history: list[OrderBookSnap] = []
        self._cancel_ratio: float = 0.0
        self._probe_cycles: int = 0
        self._last_passive_vol: float = 0.0
        self._probe_bid: bool = False
        self._probe_ask: bool = False

    def update_orderbook(self, snap: OrderBookSnap) -> None:
        self._last_snap = snap
        self._snap_history.append(snap)
        if len(self._snap_history) > 500:
            self._snap_history = self._snap_history[-500:]

    def update_trade(self, trade: Trade) -> None:
        self._flow.ingest_trade(trade)
        self._micro.ingest_trade(trade)

    def classify(self, snap: OrderBookSnap, recent_trades: list[Trade]) -> IntentSignal:
        """
        Full classification cycle.
        Call with current orderbook snap + last N trades.
        """
        now = snap.ts_ms
        metrics = self._flow.ingest_orderbook(snap)
        mp_state = self._micro.ingest_orderbook(snap)

        details = {**metrics, **{
            "micro_price": mp_state.micro_price,
            "mid_price": mp_state.mid_price,
            "imbalance": mp_state.imbalance,
            "cancel_ratio": self._cancel_ratio,
            "probe_cycles": self._probe_cycles,
            "snap_spread_bps": snap.spread_bps,
        }}

        # ── 1. MICRO_DRIFT ────────────────────────────────────────────────
        if self._micro.is_micro_drift:
            drift_bps = abs(mp_state.micro_price - mp_state.mid_price) / mp_state.mid_price * 10_000
            conf = min(drift_bps / 20.0, 1.0)   # 20bps = max confidence
            return self._signal(Intent.MICRO_DRIFT, SignalStrength.MODERATE, conf, now, snap, details)

        # ── 2. ABSORPTION ────────────────────────────────────────────────
        if self._detect_absorption(metrics, recent_trades, snap):
            return self._signal(Intent.ABSORPTION, SignalStrength.STRONG, 0.80, now, snap, details)

        # ── 3. DISTRIBUTION ──────────────────────────────────────────────
        if self._detect_distribution(metrics, recent_trades, snap):
            return self._signal(Intent.DISTRIBUTION, SignalStrength.STRONG, 0.80, now, snap, details)

        # ── 4. BULL_TRAP ────────────────────────────────────────────────
        if self._detect_bull_trap(snap, recent_trades, details):
            return self._signal(Intent.BULL_TRAP, SignalStrength.MODERATE, 0.70, now, snap, details)

        # ── 5. BEAR_TRAP ────────────────────────────────────────────────
        if self._detect_bear_trap(snap, recent_trades, details):
            return self._signal(Intent.BEAR_TRAP, SignalStrength.MODERATE, 0.70, now, snap, details)

        # ── 6. LIQUIDITY_PROBE ──────────────────────────────────────────
        if self._detect_liquidity_probe(snap, details):
            return self._signal(Intent.LIQUIDITY_PROBE, SignalStrength.WEAK, 0.60, now, snap, details)

        return IntentSignal(Intent.NEUTRAL, SignalStrength.WEAK, 1.0, now, snap.symbol, details, snap.mid_price)

    # ── Detection helpers ─────────────────────────────────────────────────

    def _detect_absorption(
        self, metrics: dict, trades: list[Trade], snap: OrderBookSnap
    ) -> bool:
        """主动买入↑ + 价格不跟涨 + high absorption ratio."""
        if not trades:
            return False
        recent = trades[-20:]
        aggressive_buys = [t for t in recent if t.side == "buy" and t.is_aggressive]
        if len(aggressive_buys) < 5:
            return False

        # price change
        prices = [t.price for t in recent]
        price_chg = (prices[-1] - prices[0]) / prices[0] * 10_000  # bps

        return (
            metrics["absorption_ratio"] > self.t.abs_absorption_min
            and metrics["order_imbalance"] > 0.3
            and price_chg < self.t.abs_price_lag_max_bps
        )

    def _detect_distribution(
        self, metrics: dict, trades: list[Trade], snap: OrderBookSnap
    ) -> bool:
        """主动卖出↑ + 价格不跟跌 + low absorption ratio."""
        if not trades:
            return False
        recent = trades[-20:]
        aggressive_sells = [t for t in recent if t.side == "sell" and t.is_aggressive]
        if len(aggressive_sells) < 5:
            return False

        prices = [t.price for t in recent]
        price_chg = (prices[-1] - prices[0]) / prices[0] * 10_000

        return (
            metrics["absorption_ratio"] < self.t.dist_absorption_max
            and metrics["order_imbalance"] < -0.3
            and price_chg > -self.t.abs_price_lag_max_bps
        )

    def _detect_bull_trap(
        self, snap: OrderBookSnap, trades: list[Trade], details: dict
    ) -> bool:
        """
        被动卖单(ask)堆积 + 突然撤单(大量卖单消失) → 准备拉升砸盘。
        Proxy: ask-side passive volume high + mid price jumps without buy aggression.
        """
        if len(self._snap_history) < 10:
            return False

        past = self._snap_history[-10:-1]
        current_ask_vol = sum(l.size for l in snap.asks[:5])
        past_ask_vols = [sum(l.size for l in s.asks[:5]) for s in past]

        if not past_ask_vols:
            return False

        avg_past = np.mean(past_ask_vols)
        ratio = current_ask_vol / avg_past if avg_past > 0 else 1.0

        # spike in ask depth → large passive sell wall building
        if ratio < 1.5:
            return False

        # then mid price pops without buy aggression
        recent = trades[-10:]
        if not recent:
            return False
        buy_agg = sum(t.size for t in recent if t.side == "buy" and t.is_aggressive)
        price_jump = (snap.mid_price - past[-1].mid_price) / past[-1].mid_price * 10_000

        return buy_agg < np.mean([t.size for t in recent]) and price_jump > 5

    def _detect_bear_trap(
        self, snap: OrderBookSnap, trades: list[Trade], details: dict
    ) -> bool:
        """
        被动买单(bid)堆积 + 突然撤单 → 准备砸盘拉抬。
        Proxy: bid-side passive volume high + mid price drops without sell aggression.
        """
        if len(self._snap_history) < 10:
            return False

        past = self._snap_history[-10:-1]
        current_bid_vol = sum(l.size for l in snap.bids[:5])
        past_bid_vols = [sum(l.size for l in s.bids[:5]) for s in past]

        if not past_bid_vols:
            return False

        avg_past = np.mean(past_bid_vols)
        ratio = current_bid_vol / avg_past if avg_past > 0 else 1.0

        if ratio < 1.5:
            return False

        recent = trades[-10:]
        if not recent:
            return False
        sell_agg = sum(t.size for t in recent if t.side == "sell" and t.is_aggressive)
        price_drop = (snap.mid_price - past[-1].mid_price) / past[-1].mid_price * 10_000

        return sell_agg < np.mean([t.size for t in recent]) and price_drop < -5

    def _detect_liquidity_probe(self, snap: OrderBookSnap, details: dict) -> bool:
        """
        大单挂而不成交 + 反复撤挂 → 流动性测试。
        Proxy: large passive orders appear in history, some get cancelled.
        We approximate by detecting large passive orders appearing then shrinking.
        """
        if len(self._snap_history) < 5:
            return False

        # look at size change of top-of-book passive orders over last N snaps
        sizes = [snap.bids[0].size if snap.bids else 0 for snap in self._snap_history[-5:]]
        # large then small then large = repeated probe
        if len(sizes) < 5:
            return False

        pattern = sizes
        large_thresh = np.mean(sizes) * 3
        large_count = sum(1 for s in pattern if s > large_thresh)
        small_count = sum(1 for s in pattern if s < large_thresh * 0.3)

        # repeated appearance/disappearance
        if large_count >= 2 and small_count >= 2:
            self._probe_cycles += 1
        else:
            self._probe_cycles = max(0, self._probe_cycles - 1)

        return self._probe_cycles >= self.t.probe_repeat_min

    # ── Signal builder ───────────────────────────────────────────────────

    def _signal(
        self,
        intent: Intent,
        strength: SignalStrength,
        confidence: float,
        ts_ms: int,
        snap: OrderBookSnap,
        details: dict,
    ) -> IntentSignal:
        return IntentSignal(
            intent=intent,
            strength=strength,
            confidence=confidence,
            ts_ms=ts_ms,
            symbol=snap.symbol,
            details=details,
            price_at_signal=snap.mid_price,
        )


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def bps_to_pct(bps: float) -> str:
    return f"{bps:.2f}bps"


def format_signal(s: IntentSignal) -> str:
    emoji = {
        Intent.BULL_TRAP: "🟡",
        Intent.BEAR_TRAP: "🔵",
        Intent.ABSORPTION: "🟢",
        Intent.DISTRIBUTION: "🔴",
        Intent.LIQUIDITY_PROBE: "⚪",
        Intent.MICRO_DRIFT: "◐",
        Intent.NEUTRAL: "⚫",
    }
    return (
        f"{emoji.get(s.intent,'⚫')} [{s.strength.value.upper()}] {s.intent.value} "
        f"| confidence={s.confidence:.0%} | price={s.price_at_signal:.4f}"
    )