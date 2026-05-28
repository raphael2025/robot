"""
Unit Tests — adapters/exchange.py
================================
Tests: message parsing for Binance, Bybit, OKX adapters.
Mock WebSocket / REST responses to verify normalisation.
"""

import gzip
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.adapters.exchange import (
    BinanceAdapter,
    BybitAdapter,
    OKXAdapter,
    get_adapter,
    ExchangeAdapter,
)


# ─────────────────────────────────────────────
#  Binance Tests
# ─────────────────────────────────────────────

class TestBinanceAdapter:
    @pytest.fixture
    def adapter(self):
        return BinanceAdapter()

    @pytest.fixture
    def binance_depth_msg(self):
        return {
            "stream": "btcusdt@depth@100ms",
            "data": {
                "E": int(time.time() * 1000),
                "s": "BTCUSDT",
                "bids": [["100.0", "5.0"], ["99.9", "3.0"]],
                "asks": [["101.0", "4.0"], ["101.1", "2.0"]],
            },
        }

    @pytest.fixture
    def binance_trade_msg(self):
        return {
            "stream": "btcusdt@trade",
            "data": {
                "E": int(time.time() * 1000),
                "s": "BTCUSDT",
                "p": "100.0",
                "q": "1.5",
                "m": False,   # m=false → buyer is taker → aggressive buy
                "T": int(time.time() * 1000),
            },
        }

    def test_parse_depth(self, adapter, binance_depth_msg):
        snaps = adapter._parse_message(json.dumps(binance_depth_msg))
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap.exchange == "binance"
        assert snap.symbol == "btcusdt"
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0
        assert len(snap.bids) == 2
        assert len(snap.asks) == 2

    def test_parse_trade(self, adapter, binance_trade_msg):
        snaps = adapter._parse_message(json.dumps(binance_trade_msg))
        assert len(snaps) == 1
        trade = snaps[0]
        # m=False → taker buy = aggressive buy
        assert trade.side == "buy"
        assert trade.is_aggressive is True
        assert trade.price == 100.0
        assert trade.size == 1.5

    def test_parse_trade_maker_sell(self, adapter):
        msg = {
            "stream": "btcusdt@trade",
            "data": {
                "E": int(time.time() * 1000),
                "s": "BTCUSDT",
                "p": "99.5",
                "q": "0.5",
                "m": True,   # maker = passive sell
                "T": int(time.time() * 1000),
            },
        }
        snaps = adapter._parse_message(json.dumps(msg))
        trade = snaps[0]
        # m=True → maker = passive sell
        assert trade.side == "sell"
        assert trade.is_aggressive is False

    def test_parse_unknown_stream(self, adapter):
        snaps = adapter._parse_message(json.dumps({"stream": "unknown", "data": {}}))
        assert snaps == []

    def test_get_ws_url(self, adapter):
        url = adapter.get_ws_url("BTCUSDT")
        assert "stream.binance.com" in url
        assert "btcusdt" in url

    def test_gzip_message(self, adapter, binance_depth_msg):
        raw = gzip.compress(json.dumps(binance_depth_msg).encode())
        snaps = adapter._parse_message(raw)
        assert len(snaps) == 1


# ─────────────────────────────────────────────
#  Bybit Tests
# ─────────────────────────────────────────────

class TestBybitAdapter:
    @pytest.fixture
    def adapter(self):
        return BybitAdapter()

    @pytest.fixture
    def bybit_depth_msg(self):
        return {
            "topic": "depth.BTCUSDT",
            "data": {
                "ts": int(time.time() * 1000),
                "b": [["100.0", "5.0"], ["99.9", "3.0"]],
                "a": [["101.0", "4.0"]],
            },
        }

    @pytest.fixture
    def bybit_trade_msg(self):
        return {
            "topic": "trade.BTCUSDT",
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "p": "100.0",
                    "v": "1.5",
                    "s": "Buy",
                    "S": "Buy",
                    "ts": int(time.time() * 1000),
                }
            ],
        }

    def test_parse_depth(self, adapter, bybit_depth_msg):
        snaps = adapter._parse_message(json.dumps(bybit_depth_msg))
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap.exchange == "bybit"
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0

    def test_parse_trade(self, adapter, bybit_trade_msg):
        snaps = adapter._parse_message(json.dumps(bybit_trade_msg))
        assert len(snaps) == 1
        trade = snaps[0]
        assert trade.side == "buy"
        assert trade.price == 100.0

    def test_get_ws_url(self, adapter):
        url = adapter.get_ws_url("BTCUSDT")
        assert "stream.bybit.com" in url

    def test_zero_size_filtered(self, adapter):
        msg = {
            "topic": "depth.BTCUSDT",
            "data": {
                "ts": int(time.time() * 1000),
                "b": [["100.0", "0.0"], ["99.9", "3.0"]],
                "a": [["101.0", "4.0"]],
            },
        }
        snaps = adapter._parse_message(json.dumps(msg))
        snap = snaps[0]
        # zero-size bid should be filtered
        assert all(b.size > 0 for b in snap.bids)


# ─────────────────────────────────────────────
#  OKX Tests
# ─────────────────────────────────────────────

class TestOKXAdapter:
    @pytest.fixture
    def adapter(self):
        return OKXAdapter()

    @pytest.fixture
    def okx_depth_msg(self):
        return {
            "args": [{"channel": "books-l1-tbt", "instId": "BTC-USDT"}],
            "data": [
                {
                    "ts": int(time.time() * 1000),
                    "bids": [["100.0", "5.0", "0", "0"], ["99.9", "3.0", "0", "0"]],
                    "asks": [["101.0", "4.0", "0", "0"]],
                }
            ],
        }

    @pytest.fixture
    def okx_trade_msg(self):
        return {
            "args": [{"channel": "trades", "instId": "BTC-USDT"}],
            "data": [
                {
                    "instId": "BTC-USDT",
                    "px": "100.0",
                    "sz": "1.5",
                    "side": "buy",
                    "ts": int(time.time() * 1000),
                }
            ],
        }

    def test_parse_depth(self, adapter, okx_depth_msg):
        snaps = adapter._parse_message(json.dumps(okx_depth_msg))
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap.exchange == "okx"
        assert snap.symbol == "BTC-USDT"
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0

    def test_parse_trade(self, adapter, okx_trade_msg):
        snaps = adapter._parse_message(json.dumps(okx_trade_msg))
        assert len(snaps) == 1
        trade = snaps[0]
        assert trade.side == "buy"
        assert trade.is_aggressive is True

    def test_pong_filtered(self, adapter):
        msg = {"op": "pong"}
        snaps = adapter._parse_message(json.dumps(msg))
        assert snaps == []


# ─────────────────────────────────────────────
#  Adapter Factory
# ─────────────────────────────────────────────

class TestAdapterFactory:
    def test_get_binance(self):
        a = get_adapter("binance")
        assert isinstance(a, BinanceAdapter)

    def test_get_bybit(self):
        a = get_adapter("bybit")
        assert isinstance(a, BybitAdapter)

    def test_get_okx(self):
        a = get_adapter("okx")
        assert isinstance(a, OKXAdapter)

    def test_unknown_exchange(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            get_adapter("kucoin")


# ─────────────────────────────────────────────
#  REST Fetch (mocked)
# ─────────────────────────────────────────────

class TestBinanceRest:
    @pytest.mark.asyncio
    async def test_fetch_orderbook_rest(self):
        """
        Verify REST orderbook fetch without aiohttp patch.
        We test the parsing logic directly by providing parsed dict data.
        The actual HTTP round-trip is tested separately in integration.
        """
        adapter = BinanceAdapter()
        raw = {
            "bids": [["100.0", "5.0"], ["99.9", "3.0"]],
            "asks": [["101.0", "4.0"]],
        }
        snap = adapter._parse_depth_rest(raw, "BTCUSDT")
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0
        assert len(snap.bids) == 2
        assert len(snap.asks) == 1