"""
Exchange Adapters — WebSocket/REST clients for Binance, Bybit, OKX.
Each adapter normalises raw exchange messages → unified OrderBookSnap / Trade.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from src.core.orderflow import OrderBookLevel, OrderBookSnap, Trade

log = logging.getLogger("adapter")


# ─────────────────────────────────────────────
#  Base Adapter
# ─────────────────────────────────────────────

@dataclass
class ExchangeConfig:
    name: str
    ws_url: str
    rest_url: str
    max_depth: int = 20
    ping_interval_ms: int = 20_000
    ping_timeout_ms: int = 10_000
    compression: str = "gzip"


class ExchangeAdapter(ABC):
    """
    Abstract base for exchange WebSocket adapters.

    Subclasses implement:
      - `get_ws_url(symbol)` → WebSocket URI
      - `_parse_message(raw)` → list[OrderBookSnap | Trade]
      - `_subscribe(ws, symbol)` → None (send subscription frames)

    The adapter exposes:
      - `stream(symbol, on_orderbook, on_trade)` — async generator
      - `fetch_orderbook_rest(symbol)` → OrderBookSnap (one-shot REST)
    """

    config: ExchangeConfig

    def __init__(self, max_orderbook_levels: int = 20):
        self.max_levels = max_orderbook_levels
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False

    @abstractmethod
    def get_ws_url(self, symbol: str) -> str:
        ...

    @abstractmethod
    def _subscribe(self, ws: websockets.WebSocketClientProtocol, symbol: str) -> None:
        ...

    @abstractmethod
    def _parse_message(self, raw: bytes | str) -> list[OrderBookSnap | Trade]:
        ...

    async def connect(self, symbol: str):
        url = self.get_ws_url(symbol)
        log.info(f"[{self.config.name}] Connecting WS → {url}")
        self._ws = await websockets.connect(
            url,
            ping_interval=self.config.ping_interval_ms / 1000,
            ping_timeout=self.config.ping_timeout_ms / 1000,
            compression=gzip if self.config.compression == "gzip" else None,
        )
        await self._subscribe(self._ws, symbol)
        self._running = True
        log.info(f"[{self.config.name}] Connected & subscribed to {symbol}")

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def stream(
        self,
        symbol: str,
        on_orderbook: Optional[callable] = None,
        on_trade: Optional[callable] = None,
    ):
        """
        Consume WebSocket messages forever, dispatching to callbacks.
        Returns an async generator yielding (OrderBookSnap | Trade).
        """
        if self._ws is None:
            await self.connect(symbol)

        async for raw in self._ws:
            events = self._parse_message(raw)
            for ev in events:
                if isinstance(ev, OrderBookSnap):
                    if on_orderbook:
                        on_orderbook(ev)
                elif isinstance(ev, Trade):
                    if on_trade:
                        on_trade(ev)
                yield ev

    @abstractmethod
    async def fetch_orderbook_rest(self, symbol: str) -> OrderBookSnap:
        ...


# ─────────────────────────────────────────────
#  Binance Adapter
# ─────────────────────────────────────────────

class BinanceAdapter(ExchangeAdapter):
    """
    Binance WebSocket API
    ──────────────────────
    Combined stream: <symbol>@depth@100ms + <symbol>@trade

    REST fallback: GET /api/v3/depth?symbol=...&limit=20

    WebSocket URL: wss://stream.binance.com:9443/stream

    Combined stream params:
        streams: <symbol>@depth@100ms/<symbol>@trade
    """

    config = ExchangeConfig(
        name="binance",
        ws_url="wss://stream.binance.com:9443/stream",
        rest_url="https://api.binance.com",
    )

    def get_ws_url(self, symbol: str) -> str:
        s = symbol.lower()
        # combined stream
        return f"{self.config.ws_url}?streams={s}@depth@100ms/{s}@trade"

    def _subscribe(self, ws: websockets.WebSocketClientProtocol, symbol: str) -> None:
        # combined streams auto-subscribe via URL param
        pass

    def _parse_message(self, raw: bytes | str) -> list[OrderBookSnap | Trade]:
        try:
            if isinstance(raw, bytes):
                raw = gzip.decompress(raw).decode()
            msg = json.loads(raw)
        except Exception:
            return []

        results: list[OrderBookSnap | Trade] = []
        stream = msg.get("stream", "")
        data = msg.get("data", {})

        if "depth" in stream:
            results.append(self._parse_depth(data, symbol=stream.split("@")[0]))
        elif "trade" in stream:
            results.append(self._parse_trade(data))
        return results

    def _parse_depth(self, data: dict, symbol: str) -> OrderBookSnap:
        bids = [
            OrderBookLevel(price=float(p), size=float(q))
            for p, q in data.get("bids", [])[: self.max_levels]
        ]
        asks = [
            OrderBookLevel(price=float(p), size=float(q))
            for p, q in data.get("asks", [])[: self.max_levels]
        ]
        return OrderBookSnap(
            exchange="binance",
            symbol=symbol,
            ts_ms=int(data.get("E", time.time() * 1000)),
            bids=bids,
            asks=asks,
        )

    def _parse_trade(self, data: dict) -> Trade:
        return Trade(
            exchange="binance",
            symbol=data.get("s", ""),
            ts_ms=int(data.get("T", time.time() * 1000)),
            price=float(data.get("p", 0)),
            size=float(data.get("q", 0)),
            side="buy" if not data.get("m", True) else "sell",  # m=true → maker = passive sell
            is_aggressive=not data.get("m", True),               # m=false → taker = aggressive
        )

    async def fetch_orderbook_rest(self, symbol: str) -> OrderBookSnap:
        import aiohttp
        url = f"{self.config.rest_url}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": self.max_levels}
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()
        return self._parse_depth_rest(data, symbol)

    def _parse_depth_rest(self, data: dict, symbol: str) -> OrderBookSnap:
        bids = [
            OrderBookLevel(price=float(p), size=float(q))
            for p, q in data.get("bids", [])[: self.max_levels]
        ]
        asks = [
            OrderBookLevel(price=float(p), size=float(q))
            for p, q in data.get("asks", [])[: self.max_levels]
        ]
        return OrderBookSnap(
            exchange="binance",
            symbol=symbol,
            ts_ms=int(time.time() * 1000),
            bids=bids,
            asks=asks,
        )


# ─────────────────────────────────────────────
#  Bybit Adapter
# ─────────────────────────────────────────────

class BybitAdapter(ExchangeAdapter):
    """
    Bybit WebSocket API v2
    ───────────────────────
    Public channels:
      - depth.{symbol}        (orderbook, 100ms)
      - trade.{symbol}       (trades)

    ws URL: wss://stream.bybit.com/v5/public/spot

    Auth: none for public. For USDT-perp: wss://stream.bybit.com/v5/public/linear
    """

    config = ExchangeConfig(
        name="bybit",
        ws_url="wss://stream.bybit.com/v5/public/spot",
        rest_url="https://api.bybit.com",
    )

    def get_ws_url(self, symbol: str) -> str:
        # We'll subscribe to both depth + trade in _subscribe
        return self.config.ws_url

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol, symbol: str) -> None:
        for ch in ["depth", "trade"]:
            msg = {
                "op": "subscribe",
                "args": [f"{ch}.{symbol.upper()}"],
            }
            await ws.send(json.dumps(msg))
            log.info(f"[bybit] Subscribed → {ch}.{symbol.upper()}")

    def _parse_message(self, raw: bytes | str) -> list[OrderBookSnap | Trade]:
        try:
            if isinstance(raw, bytes):
                raw = gzip.decompress(raw).decode()
            msg = json.loads(raw)
        except Exception:
            return []

        results: list[OrderBookSnap | Trade] = []
        topic = msg.get("topic", "")

        if topic.startswith("depth."):
            d = msg.get("data", {})
            results.append(self._parse_depth(d, topic.split(".")[-1]))
        elif topic.startswith("trade."):
            for t in msg.get("data", []):
                results.append(self._parse_trade(t))
        return results

    def _parse_depth(self, data: dict, symbol: str) -> OrderBookSnap:
        bids = [
            OrderBookLevel(price=float(p), size=float(float(s)))
            for p, s in data.get("b", [])[: self.max_levels]
            # filter zero-size
            if float(s) > 0
        ]
        asks = [
            OrderBookLevel(price=float(p), size=float(float(s)))
            for p, s in data.get("a", [])[: self.max_levels]
            if float(s) > 0
        ]
        return OrderBookSnap(
            exchange="bybit",
            symbol=symbol,
            ts_ms=int(data.get("ts", time.time() * 1000)),
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _parse_trade(data: dict) -> Trade:
        return Trade(
            exchange="bybit",
            symbol=data.get("symbol", ""),
            ts_ms=int(data.get("ts", time.time() * 1000)),
            price=float(data.get("p", 0)),
            size=float(data.get("v", 0)),
            side="buy" if data.get("s", "") == "Buy" else "sell",
            is_aggressive=data.get("S", "") in ("Buy", "Sell"),
        )

    async def fetch_orderbook_rest(self, symbol: str) -> OrderBookSnap:
        import aiohttp
        url = f"{self.config.rest_url}/v5/market/orderbook"
        params = {"category": "spot", "symbol": symbol.upper(), "limit": self.max_levels}
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()
        d = data.get("result", {})
        return OrderBookSnap(
            exchange="bybit",
            symbol=symbol,
            ts_ms=int(d.get("ts", time.time() * 1000)),
            bids=[
                OrderBookLevel(price=float(p), size=float(s))
                for p, s in d.get("b", [])[: self.max_levels]
            ],
            asks=[
                OrderBookLevel(price=float(p), size=float(s))
                for p, s in d.get("a", [])[: self.max_levels]
            ],
        )


# ─────────────────────────────────────────────
#  OKX Adapter
# ─────────────────────────────────────────────

class OKXAdapter(ExchangeAdapter):
    """
    OKX WebSocket API v5
    ─────────────────────
    Channels:
      - instruments:<symbol>  (not needed for us)
      - books-l1-tbt:<symbol>  (orderbook, top-best-1, triggered-by-top)
      - trades:<symbol>        (trades)

    ws URL: wss://ws.okx.com:8443/ws/v5/public (spot)
            wss://ws.okx.com:8443/ws/v5/business (business channels)

    REST: GET /api/v5/market/books-l1?instId=<symbol>&ll=20
    """

    config = ExchangeConfig(
        name="okx",
        ws_url="wss://ws.okx.com:8443/ws/v5/public",
        rest_url="https://www.okx.com",
    )

    def get_ws_url(self, symbol: str) -> str:
        return self.config.ws_url

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol, symbol: str) -> None:
        # OKX needs explicit subscribe op
        for ch in ["books-l1-tbt", "trades"]:
            msg = {
                "op": "subscribe",
                "args": [
                    {
                        "channel": ch,
                        "instId": symbol.upper(),
                    }
                ],
            }
            await ws.send(json.dumps(msg))
            log.info(f"[okx] Subscribed → {ch} {symbol.upper()}")

    def _parse_message(self, raw: bytes | str) -> list[OrderBookSnap | Trade]:
        try:
            if isinstance(raw, bytes):
                raw = gzip.decompress(raw).decode()
            msg = json.loads(raw)
        except Exception:
            return []

        # pong
        if msg.get("op") == "pong":
            return []

        results: list[OrderBookSnap | Trade] = []
        for arg in msg.get("args", []):
            data = msg.get("data", [])
            ch = arg.get("channel", "")
            if ch.startswith("books"):
                for d in data:
                    results.append(self._parse_depth(d, arg.get("instId", "")))
            elif ch == "trades":
                for t in data:
                    results.append(self._parse_trade(t))
        return results

    def _parse_depth(self, data: dict, symbol: str) -> OrderBookSnap:
        bids = [
            OrderBookLevel(price=float(p), size=float(s))
            for p, s, _, _ in data.get("bids", [])[: self.max_levels]
            if float(s) > 0
        ]
        asks = [
            OrderBookLevel(price=float(p), size=float(s))
            for p, s, _, _ in data.get("asks", [])[: self.max_levels]
            if float(s) > 0
        ]
        return OrderBookSnap(
            exchange="okx",
            symbol=symbol,
            ts_ms=int(data.get("ts", time.time() * 1000)),
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _parse_trade(data: dict) -> Trade:
        return Trade(
            exchange="okx",
            symbol=data.get("instId", ""),
            ts_ms=int(data.get("ts", time.time() * 1000)),
            price=float(data.get("px", 0)),
            size=float(data.get("sz", 0)),
            side="buy" if data.get("side", "") == "buy" else "sell",
            is_aggressive=True,  # trades are always aggressive
        )

    async def fetch_orderbook_rest(self, symbol: str) -> OrderBookSnap:
        import aiohttp
        url = f"{self.config.rest_url}/api/v5/market/books-l1"
        params = {"instId": symbol.upper(), "ll": self.max_levels}
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as r:
                r.raise_for_status()
                data = await r.json()
        d = (data.get("data", []) or [{}])[0]
        return self._parse_depth(d, symbol)


# ─────────────────────────────────────────────
#  Adapter Factory
# ─────────────────────────────────────────────

ADAPTERS = {
    "binance": BinanceAdapter,
    "bybit": BybitAdapter,
    "okx": OKXAdapter,
}


def get_adapter(exchange: str, **kwargs) -> ExchangeAdapter:
    cls = ADAPTERS.get(exchange.lower())
    if cls is None:
        raise ValueError(f"Unknown exchange: {exchange}. Supported: {list(ADAPTERS)}")
    return cls(**kwargs)