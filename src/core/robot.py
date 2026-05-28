"""
Robot Trading Strategy Runner
============================
Connects the full pipeline:
  ExchangeAdapter → OrderFlowClassifier → IntentSignal → action

Supports:
  - Live mode: WebSocket streaming
  - Backtest mode: replay from stored orderbook snapshots
  - Alert mode: print/intend to a webhook on signals
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .orderflow import (
    InstitutionalIntentClassifier,
    IntentSignal,
    MicroPriceEngine,
    OrderFlowMetrics,
    OrderBookSnap,
    Trade,
    format_signal,
)
from src.adapters.exchange import ExchangeAdapter, get_adapter

log = logging.getLogger("robot.strategy")


@dataclass
class RobotConfig:
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    adapter_kwargs: dict = field(default_factory=dict)
    classifier_kwargs: dict = field(default_factory=dict)
    # thresholds override
    signal_min_confidence: float = 0.60
    # alert callback
    alert_callback: Optional[Callable[[IntentSignal], None]] = None
    # Webhook URL for alerts
    alert_webhook: Optional[str] = None


class Robot:
    """
    Main robot class.

    Pipeline per update cycle:
        on_orderbook(snap)   → update classifier   → check MICRO_DRIFT
        on_trade(trade)      → update metrics       → classify on interval
        every `classify_interval_ms` ms → full classify()
    """

    def __init__(self, config: RobotConfig):
        self.cfg = config
        self.adapter: ExchangeAdapter = get_adapter(
            config.exchange, **config.adapter_kwargs
        )
        self.classifier = InstitutionalIntentClassifier(**config.classifier_kwargs)
        self._last_trades: list[Trade] = []
        self._last_classify_ms = 0
        self._classify_interval_ms = 500  # full classify every 500ms
        self._last_signal: Optional[IntentSignal] = None
        self._signals: list[IntentSignal] = []

    async def start_live(self):
        """Connect WebSocket and run the event loop."""
        log.info(f"Starting live robot: {self.cfg.exchange} {self.cfg.symbol}")
        await self.adapter.connect(self.cfg.symbol)

        try:
            async for event in self.adapter.stream(
                self.cfg.symbol,
                on_orderbook=self._on_orderbook,
                on_trade=self._on_trade,
            ):
                # stream is async generator — callbacks already fired above
                # yield control back to allow cancellation
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            log.info("Robot cancelled — disconnecting")
        finally:
            await self.adapter.disconnect()

    def _on_orderbook(self, snap: OrderBookSnap):
        self.classifier.update_orderbook(snap)
        # micro-drift check on every orderbook (fast path)
        mp_state = self.classifier._micro.ingest_orderbook(snap)
        if self.classifier._micro.is_micro_drift:
            sig = self.classifier.classify(snap, self._last_trades)
            self._emit(sig)
        self._last_classify_ms = snap.ts_ms

    def _on_trade(self, trade: Trade):
        self._last_trades.append(trade)
        if len(self._last_trades) > 500:
            self._last_trades = self._last_trades[-500:]
        self.classifier.update_trade(trade)

    def _emit(self, sig: IntentSignal):
        if sig.confidence < self.cfg.signal_min_confidence:
            return
        if sig.intent == sig.intent.__class__.NEUTRAL:
            return
        self._last_signal = sig
        self._signals.append(sig)
        log.warning(f"SIGNAL: {format_signal(sig)}")
        if self.cfg.alert_callback:
            self.cfg.alert_callback(sig)
        if self.cfg.alert_webhook:
            asyncio.create_task(self._webhook_alert(sig))

    async def _webhook_alert(self, sig: IntentSignal):
        import aiohttp
        payload = {
            "intent": sig.intent.value,
            "strength": sig.strength.value,
            "confidence": sig.confidence,
            "symbol": sig.symbol,
            "price": sig.price_at_signal,
            "ts_ms": sig.ts_ms,
            "details": {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in sig.details.items()},
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(self.cfg.alert_webhook, json=payload) as r:
                    log.info(f"Webhook → {r.status}")
        except Exception as e:
            log.error(f"Webhook failed: {e}")


# ─────────────────────────────────────────────
#  Backtest Runner
# ─────────────────────────────────────────────

@dataclass
class BacktestConfig:
    data_source: str = "file"          # "file" | "rest"
    data_path: str = ""               # parquet/csv path
    exchange: str = "binance"
    symbol: str = "BTCUSDT"


async def run_backtest(
    cfg: BacktestConfig,
    classifier: InstitutionalIntentClassifier,
    signals_out: bool = True,
) -> list[IntentSignal]:
    """
    Replay orderbook/trade data from file or REST calls,
    collect all signals.
    """
    import pandas as pd

    log.info(f"Backtest starting: {cfg.data_path or cfg.exchange}")
    signals: list[IntentSignal] = []
    classifier = classifier  # reuse provided instance

    if cfg.data_source == "file":
        df = pd.read_parquet(cfg.data_path)
        # assume columns: ts_ms, bids_json, asks_json, trades_json
        for _, row in df.iterrows():
            snap = OrderBookSnap(
                exchange=cfg.exchange,
                symbol=cfg.symbol,
                ts_ms=int(row["ts_ms"]),
                bids=_parse_levels(row["bids_json"]),
                asks=_parse_levels(row["asks_json"]),
            )
            classifier.update_orderbook(snap)
            sig = classifier.classify(snap, [])
            if sig.intent.value != "neutral" and sig.confidence > 0.6:
                signals.append(sig)
                if signals_out:
                    log.warning(f"BT: {format_signal(sig)}")

    await asyncio.sleep(0)
    return signals


def _parse_levels(data):
    """Parse JSON-encoded order book levels."""
    import json
    if isinstance(data, str):
        data = json.loads(data)
    from .orderflow import OrderBookLevel
    return [OrderBookLevel(price=p, size=s) for p, s in data]