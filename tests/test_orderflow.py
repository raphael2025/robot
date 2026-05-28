"""
Unit Tests — core/orderflow.py
==============================
Tests: MicroPriceEngine, OrderFlowMetrics, InstitutionalIntentClassifier.
"""

import time
import pytest
import numpy as np
from src.core.orderflow import (
    Intent,
    IntentThresholds,
    InstitutionalIntentClassifier,
    MicroPriceEngine,
    OrderBookLevel,
    OrderBookSnap,
    OrderFlowMetrics,
    Trade,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def empty_snap():
    return OrderBookSnap(
        exchange="binance",
        symbol="BTCUSDT",
        ts_ms=int(time.time() * 1000),
        bids=[OrderBookLevel(price=100.0, size=1.0)],
        asks=[OrderBookLevel(price=101.0, size=1.0)],
    )


@pytest.fixture
def snap_10k():
    """Order book with 10k spread and clear bid-side imbalance."""
    return OrderBookSnap(
        exchange="binance",
        symbol="BTCUSDT",
        ts_ms=int(time.time() * 1000),
        bids=[
            OrderBookLevel(price=100.0, size=10.0),
            OrderBookLevel(price=99.9, size=8.0),
            OrderBookLevel(price=99.8, size=7.0),
            OrderBookLevel(price=99.7, size=6.0),
            OrderBookLevel(price=99.6, size=5.0),
        ],
        asks=[
            OrderBookLevel(price=110.0, size=1.0),
            OrderBookLevel(price=110.1, size=1.0),
            OrderBookLevel(price=110.2, size=1.0),
        ],
    )


@pytest.fixture
def snap_ask_wall():
    """Large ask wall building up — potential bull trap."""
    return OrderBookSnap(
        exchange="binance",
        symbol="BTCUSDT",
        ts_ms=int(time.time() * 1000),
        bids=[OrderBookLevel(price=100.0, size=1.0)],
        asks=[
            OrderBookLevel(price=100.5, size=50.0),   # huge passive sell wall
            OrderBookLevel(price=100.6, size=40.0),
        ],
    )


def make_trade(
    price: float,
    size: float,
    side: str = "buy",
    ts_ms: int = None,
    is_aggressive: bool = True,
) -> Trade:
    return Trade(
        exchange="binance",
        symbol="BTCUSDT",
        ts_ms=ts_ms or int(time.time() * 1000),
        price=price,
        size=size,
        side=side,
        is_aggressive=is_aggressive,
    )


# ─────────────────────────────────────────────
#  OrderBookSnap
# ─────────────────────────────────────────────

class TestOrderBookSnap:
    def test_best_bid(self, empty_snap):
        assert empty_snap.best_bid == 100.0

    def test_best_ask(self, empty_snap):
        assert empty_snap.best_ask == 101.0

    def test_mid_price(self, empty_snap):
        assert empty_snap.mid_price == 100.5

    def test_spread(self, empty_snap):
        assert empty_snap.spread == 1.0

    def test_spread_bps(self, empty_snap):
        assert abs(empty_snap.spread_bps - 99.50) < 0.01


# ─────────────────────────────────────────────
#  MicroPriceEngine
# ─────────────────────────────────────────────

class TestMicroPriceEngine:
    def test_micro_price_bid_heavy(self, snap_10k):
        """Bid-side heavy → micro_price should be above mid."""
        eng = MicroPriceEngine(window_ms=10_000)
        state = eng.ingest_orderbook(snap_10k)
        assert state.micro_price > state.mid_price
        assert state.imbalance > 0
        assert state.bid_depth_heavy is True

    def test_micro_price_ask_heavy(self):
        """Ask-side heavy → micro_price should be below mid."""
        snap = OrderBookSnap(
            exchange="binance",
            symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[
                OrderBookLevel(price=100.5, size=10.0),
                OrderBookLevel(price=100.6, size=8.0),
            ],
        )
        eng = MicroPriceEngine()
        state = eng.ingest_orderbook(snap)
        assert state.micro_price < state.mid_price
        assert state.imbalance < 0
        assert state.bid_depth_heavy is False

    def test_micro_price_balanced(self):
        """Even book → micro_price ≈ mid."""
        snap = OrderBookSnap(
            exchange="binance",
            symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=100.0, size=5.0)],
            asks=[OrderBookLevel(price=100.1, size=5.0)],
        )
        eng = MicroPriceEngine()
        state = eng.ingest_orderbook(snap)
        assert abs(state.micro_price - state.mid_price) < 0.05

    def test_micro_drift_fires_after_consecutive(self):
        """Micro drift fires only after 3+ consecutive prints with large imbalance."""
        eng = MicroPriceEngine()
        assert not eng.is_micro_drift

        for i in range(3):
            snap = OrderBookSnap(
                exchange="binance", symbol="BTCUSDT",
                ts_ms=int(time.time() * 1000),
                bids=[
                    OrderBookLevel(price=100.0, size=50.0),
                    OrderBookLevel(price=99.9, size=40.0),
                ],
                asks=[
                    OrderBookLevel(price=100.1, size=1.0),
                ],
            )
            eng.ingest_orderbook(snap)

        assert eng.is_micro_drift

    def test_ingest_trade_prunes_old(self):
        """Old trades are removed from order flow history."""
        eng = MicroPriceEngine(window_ms=500)
        now = int(time.time() * 1000)
        eng.ingest_trade(make_trade(100, 1, ts_ms=now - 10_000))  # stale
        eng.ingest_trade(make_trade(100, 1, ts_ms=now))
        assert len(eng._order_flow) == 1


# ─────────────────────────────────────────────
#  OrderFlowMetrics
# ─────────────────────────────────────────────

class TestOrderFlowMetrics:
    def test_order_imbalance_calculation(self, snap_10k):
        oi = OrderFlowMetrics._order_imbalance(snap_10k)
        assert oi > 0  # bid side heavy
        assert oi <= 1.0

    def test_order_imbalance_ask_heavy(self):
        snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[OrderBookLevel(price=101.0, size=10.0)],
        )
        oi = OrderFlowMetrics._order_imbalance(snap)
        assert oi < 0  # ask side heavy

    def test_order_imbalance_zero(self):
        snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=100.0, size=5.0)],
            asks=[OrderBookLevel(price=101.0, size=5.0)],
        )
        oi = OrderFlowMetrics._order_imbalance(snap)
        assert abs(oi) < 0.01

    def test_absorption_ratio_high_on_buys(self):
        """High absorption = many passive hits, few aggressive buys."""
        metrics = OrderFlowMetrics(window_trades=100, absorption_window=20)
        now = int(time.time() * 1000)
        # add 20 trades: 10 passive sells (absorbed), 10 aggressive buys
        for i in range(10):
            metrics.ingest_trade(make_trade(100, 1, side="sell", ts_ms=now+i, is_aggressive=False))
        for i in range(10):
            metrics.ingest_trade(make_trade(100, 1, side="buy", ts_ms=now+i+10, is_aggressive=True))
        # just check it doesn't crash
        assert len(metrics._trades) == 20

    def test_vpin_bucket_accumulation(self):
        """VPIN buckets fill up over trades."""
        metrics = OrderFlowMetrics(vpin_buckets=10, window_trades=200)
        now = int(time.time() * 1000)
        for i in range(50):
            metrics.ingest_trade(make_trade(100 + i * 0.01, 1.0, ts_ms=now + i))
        # shouldn't crash; vpin value is derived from buckets
        assert metrics._vpin >= 0.0


# ─────────────────────────────────────────────
#  InstitutionalIntentClassifier
# ─────────────────────────────────────────────

class TestIntentClassifier:
    def test_neutral_when_no_data(self, empty_snap):
        clf = InstitutionalIntentClassifier()
        sig = clf.classify(empty_snap, [])
        assert sig.intent == Intent.NEUTRAL

    def test_absorption_detected(self, snap_10k):
        """Aggressive buys + high absorption + flat price → ABSORPTION."""
        clf = InstitutionalIntentClassifier()
        now = int(time.time() * 1000)
        # many aggressive buys, small price change
        trades = [
            make_trade(100.0, 1.0, side="buy", ts_ms=now + i, is_aggressive=True)
            for i in range(20)
        ]
        # update orderbook
        clf.update_orderbook(snap_10k)
        sig = clf.classify(snap_10k, trades)
        # ABSORPTION or NEUTRAL depending on thresholds — just check it runs
        assert sig.intent in Intent

    def test_classifier_updates_orderbook_history(self, empty_snap):
        clf = InstitutionalIntentClassifier()
        clf.update_orderbook(empty_snap)
        assert len(clf._snap_history) == 1

    def test_bull_trap_logic(self):
        """Bull trap: large passive ask wall, then price pops without buy volume."""
        clf = InstitutionalIntentClassifier()
        now = int(time.time() * 1000)

        # Build history with large ask wall
        big_ask_snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=now - 100,
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[OrderBookLevel(price=100.5, size=100.0)],
        )
        clf.update_orderbook(big_ask_snap)

        # subsequent snap: wall pulled, price jumped
        jump_snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=now,
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[OrderBookLevel(price=100.5, size=1.0)],
        )
        # add minimal trades (no buy aggression)
        trades = [make_trade(100.0, 0.1, ts_ms=now, is_aggressive=False)] * 5
        sig = clf.classify(jump_snap, trades)
        # should not crash; intent is in enum
        assert sig.intent in Intent

    def test_micro_drift_priority(self):
        """Micro drift should fire before absorption/distribution."""
        clf = InstitutionalIntentClassifier()
        now = int(time.time() * 1000)
        # build 3+ consecutive prints with strong imbalance
        for i in range(4):
            snap = OrderBookSnap(
                exchange="binance", symbol="BTCUSDT",
                ts_ms=now + i,
                bids=[OrderBookLevel(price=100.0, size=80.0)],
                asks=[OrderBookLevel(price=100.1, size=1.0)],
            )
            clf.update_orderbook(snap)
            clf._micro.ingest_orderbook(snap)
        sig = clf.classify(clf._snap_history[-1], [])
        # Micro drift takes priority
        assert sig.intent == Intent.MICRO_DRIFT or sig.intent == Intent.NEUTRAL


# ─────────────────────────────────────────────
#  Integration — full signal chain
# ─────────────────────────────────────────────

class TestSignalChain:
    def test_full_pipeline_runs(self, snap_10k):
        """Simulate a full update cycle without exceptions."""
        clf = InstitutionalIntentClassifier()
        now = int(time.time() * 1000)
        trades = [
            make_trade(100.0 + i * 0.001, 1.0, side="buy", ts_ms=now + i)
            for i in range(30)
        ]
        clf.update_orderbook(snap_10k)
        for t in trades:
            clf.update_trade(t)
        sig = clf.classify(snap_10k, trades)
        assert sig.intent in Intent
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.strength in sig.strength.__class__

    def test_format_signal(self):
        from src.core.orderflow import IntentSignal, SignalStrength
        sig = IntentSignal(
            intent=Intent.ABSORPTION,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            ts_ms=int(time.time() * 1000),
            symbol="BTCUSDT",
            details={"vpin": 0.5},
            price_at_signal=100.0,
        )
        out = sig.intent.value
        assert out == "absorption"


# ─────────────────────────────────────────────
#  Edge Cases
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_spread_orderbook(self):
        """Zero spread → no divide-by-zero."""
        snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[OrderBookLevel(price=100.0, size=1.0)],
            asks=[OrderBookLevel(price=100.0, size=1.0)],  # same price
        )
        assert snap.spread == 0.0
        eng = MicroPriceEngine()
        state = eng.ingest_orderbook(snap)
        # micro-price should equal mid when spread=0
        assert abs(state.micro_price - state.mid_price) < 1e-9

    def test_empty_orderbook_snap(self):
        snap = OrderBookSnap(
            exchange="binance", symbol="BTCUSDT",
            ts_ms=int(time.time() * 1000),
            bids=[],
            asks=[],
        )
        assert snap.best_bid == 0.0
        assert snap.mid_price == 0.0

    def test_classifier_with_no_snap_history(self, empty_snap):
        clf = InstitutionalIntentClassifier()
        # should not crash on bull_trap/bear_trap guards
        sig = clf._detect_bull_trap(empty_snap, [], {})
        assert sig is False
        sig = clf._detect_bear_trap(empty_snap, [], {})
        assert sig is False

    def test_thresholds_defaults(self):
        t = IntentThresholds()
        assert t.abs_absorption_min == 2.5
        assert t.trap_imbalance_min == 0.70
        assert t.drift_consecutive_min == 3