"""
orderflow_detector.py
=====================
基于你提供的原始框架实现的机构意图检测器：

    ├── 被动买单大量堆积 + 突然撤单  → 准备砸盘（诱多）
    ├── 被动卖单大量堆积 + 突然撤单  → 准备拉升（诱空）
    ├── 主动买入 + 价格不跟涨        → 暗中吸货（收集）
    ├── 主动卖出 + 价格不跟跌        → 暗中派发（分发）
    ├── 大单挂而不成交 + 反复撤挂     → 测试流动性/诱导
    └── Micro-Price 持续偏离价格      → 方向信号

数据来源: Binance WebSocket
  - <symbol>@aggTrade    (逐笔成交，主动/被动)
  - <symbol>@depth20@100ms (订单簿20档，100ms更新)
"""
import asyncio
import json
import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import aiohttp


# ─────────────────────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────────────────────

class Intent(Enum):
    BULL_TRAP   = "bull_trap"     # 被动卖单堆积 + 撤单 → 准备拉升后砸
    BEAR_TRAP   = "bear_trap"     # 被动买单堆积 + 撤单 → 准备砸盘后拉
    ACCUMULATING = "accumulating" # 主动买入 + 价格不跟涨
    DISTRIBUTING = "distributing" # 主动卖出 + 价格不跟跌
    LIQUIDITY_PROBE = "liquidity_probe"  # 大单挂而不成交/反复撤挂
    MICRO_DRIFT  = "micro_drift"  # Micro-Price 持续偏离
    NEUTRAL      = "neutral"


@dataclass
class OrderBookLevel:
    price: float
    size: float
    order_count: int = 1


@dataclass
class OrderBookState:
    bids: dict[float, float] = field(default_factory=dict)   # price → size
    asks: dict[float, float] = field(default_factory=dict)   # price → size
    last_update_id: int = 0
    ts: float = 0.0  # local timestamp of last update


@dataclass
class TradeEvent:
    price: float
    size: float
    side: str       # "buy"=taker-buy(主动买入), "sell"=taker-sell(主动卖出)
    is_buyer_maker: bool  # True = seller was taker (aggressive seller), False = buyer was taker
    ts: int         # trade timestamp ms
    trade_id: int


@dataclass
class IntentSignal:
    intent: Intent
    confidence: float          # 0-1
    price: float
    ts: int
    metadata: dict


# ─────────────────────────────────────────────────────────────
#  订单簿分析器
# ─────────────────────────────────────────────────────────────

class OrderBookAnalyzer:
    """
    分析订单簿，检测被动单堆积和撤单信号。
    """

    def __init__(self, depth_levels: int = 20):
        self.depth_levels = depth_levels

        # 订单簿状态
        self.ob = OrderBookState()

        # 历史版本（用于检测撤单）
        self.ob_history: deque[dict] = deque(maxlen=20)  # keep last 20 snapshots

        # 被动单堆积阈值
        self.passive_size_threshold: float = 5.0    # BTC，5个以上算大量
        self.passive_count_threshold: int = 3      # 3个以上挂单价

        # Micro-Price
        self.micro_price: float = 0.0
        self.micro_price_history: deque[float] = deque(maxlen=60)

        # 上一帧快照（用于检测撤单）
        self._prev_bids: dict[float, float] = {}
        self._prev_asks: dict[float, float] = {}

    def update_from_depth_msg(self, bids_raw, asks_raw, update_id: int):
        """从 Binance depth WebSocket 消息更新订单簿."""
        self.ob.bids = {float(p): float(s) for p, s in bids_raw}
        self.ob.asks = {float(p): float(s) for p, s in asks_raw}
        self.ob.last_update_id = update_id
        self.ob.ts = time.time()

        # 检测撤单（之前有，现在没了或变小了）
        self._detect_cancels()

        # 存储历史
        self.ob_history.append({
            'bids': dict(self.ob.bids),
            'asks': dict(self.ob.asks),
            'ts': self.ob.ts,
        })

    def _detect_cancels(self):
        """检测被动单撤单（从上一帧到当前帧消失的单子）."""
        # 被动卖单撤单：之前在asks里有的价格，现在没了或变小了
        for price, prev_size in list(self._prev_asks.items()):
            if price not in self.ob.asks:
                # 完整撤单
                self._record_cancel(side='ask', price=price, size=prev_size)
            elif self.ob.asks[price] < prev_size:
                # 部分撤单
                self._record_cancel(side='ask', price=price, size=prev_size - self.ob.asks[price])

        # 被动买单撤单
        for price, prev_size in list(self._prev_bids.items()):
            if price not in self.ob.bids:
                self._record_cancel(side='bid', price=price, size=prev_size)
            elif self.ob.bids[price] < prev_size:
                self._record_cancel(side='bid', price=price, size=prev_size - self.ob.bids[price])

        self._prev_bids = dict(self.ob.bids)
        self._prev_asks = dict(self.ob.asks)

    # 撤单记录（用于信号检测）
    def _record_cancel(self, side: str, price: float, size: float):
        """记录一次撤单事件."""
        # 存储到 instance 变量，供上层检测
        self._last_cancel = {'side': side, 'price': price, 'size': size, 'ts': time.time()}

    # Micro-Price 计算
    def compute_micro_price(self) -> float:
        """
        Micro-Price = w * best_bid + (1-w) * best_ask
        w = bid_size / (bid_size + ask_size)
        best_bid = highest bid (max) = closest to mid from below
        best_ask = lowest ask  (min) = closest to mid from above
        """
        best_bid_price = max(self.ob.bids.keys()) if self.ob.bids else 0
        best_ask_price = min(self.ob.asks.keys()) if self.ob.asks else 0
        best_bid_size  = self.ob.bids.get(best_bid_price, 0)
        best_ask_size  = self.ob.asks.get(best_ask_price, 0)
        total = best_bid_size + best_ask_size + 1e-9
        w = best_bid_size / total
        return w * best_bid_price + (1 - w) * best_ask_price

    # ── 被动单堆积检测 ───────────────────────────────────────────────
    def get_passive_pressure(self) -> dict:
        """
        返回被动买卖压力。
        被动卖单 = best_ask 以上的ask（价格更高，等着卖）
        被动买单 = best_bid 以下的bid（价格更低，等着买）
        best_bid = max(bids) = highest bid = closest to mid from below
        best_ask = min(asks) = lowest ask  = closest to mid from above
        """
        best_bid = max(self.ob.bids.keys()) if self.ob.bids else 0
        best_ask = min(self.ob.asks.keys()) if self.ob.asks else 0

        passive_bid_size = sum(
            size for price, size in self.ob.bids.items()
            if price < best_bid  # below best bid
        )
        passive_ask_size = sum(
            size for price, size in self.ob.asks.items()
            if price > best_ask  # above best ask
        )

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid,
            'passive_bid_size': passive_bid_size,
            'passive_ask_size': passive_ask_size,
            'passive_bid_count': sum(1 for p in self.ob.bids if p < best_bid),
            'passive_ask_count': sum(1 for p in self.ob.asks if p > best_ask),
        }

    def get_imbalance(self) -> float:
        """订单簿失衡: (bid - ask) / (bid + ask)"""
        bid_total = sum(self.ob.bids.values())
        ask_total = sum(self.ob.asks.values())
        total = bid_total + ask_total + 1e-9
        return (bid_total - ask_total) / total

    def get_large_passive_orders(self, side: str = 'ask', min_size: float = 5.0) -> list:
        """获取某侧大量 passive 订单（高于best_ask的ask，或低于best_bid的bid）。"""
        if side == 'ask':
            best_ask = min(self.ob.asks.keys()) if self.ob.asks else 0  # lowest ask = touched
            return [(p, s) for p, s in self.ob.asks.items() if s >= min_size and p > best_ask]
        else:
            best_bid = max(self.ob.bids.keys()) if self.ob.bids else 0  # highest bid = touched
            return [(p, s) for p, s in self.ob.bids.items() if s >= min_size and p < best_bid]

    @property
    def last_cancel(self) -> dict:
        return getattr(self, '_last_cancel', {'side': None, 'ts': 0})


# ─────────────────────────────────────────────────────────────
#  成交流分析器
# ─────────────────────────────────────────────────────────────

class TradeFlowAnalyzer:
    """
    分析成交流，检测主动买卖 vs 价格变动关系。
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.trades: deque[TradeEvent] = deque(maxlen=1000)

        # 主动成交方向
        self.buy_volume = 0.0   # taker-buy (主动买)
        self.sell_volume = 0.0  # taker-sell (主动卖)

        self.price_history: deque[float] = deque(maxlen=window)
        self.volume_history: deque[float] = deque(maxlen=window)
        self.buy_ratio_history: deque[float] = deque(maxlen=window)

    def add_trade(self, trade: TradeEvent):
        self.trades.append(trade)

        if trade.is_buyer_maker:
            # seller was taker = aggressive seller → push price down
            self.sell_volume += trade.size
        else:
            # buyer was taker = aggressive buyer → push price up
            self.buy_volume += trade.size

        self.price_history.append(trade.price)
        self.volume_history.append(trade.size)

        # buy ratio (rolling)
        total = self.buy_volume + self.sell_volume + 1e-9
        br = self.buy_volume / total
        self.buy_ratio_history.append(br)

    def get_metrics(self) -> dict:
        """返回当前成交流指标."""
        total = self.buy_volume + self.sell_volume + 1e-9
        buy_ratio = self.buy_volume / total

        # OI = Order Imbalance
        oi = (buy_ratio - 0.5) * 2

        # VPIN (Volume-synchronized Probability of Informed Trading)
        # VPIN = |buy - sell| / (buy + sell)  per bucket, then average
        vpin = abs(self.buy_volume - self.sell_volume) / total

        # Price momentum (last N trades)
        prices = list(self.price_history)
        if len(prices) >= 5:
            recent_ret = (prices[-1] - prices[-5]) / prices[-5] * 100  # % return
        else:
            recent_ret = 0.0

        # Volume spike
        vol_avg = statistics.mean(self.volume_history) if self.volume_history else 0
        vol_current = self.volume_history[-1] if self.volume_history else 0
        vol_ratio = vol_current / (vol_avg + 1e-9)

        return {
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'buy_ratio': buy_ratio,
            'oi': oi,
            'vpin': vpin,
            'recent_ret_pct': recent_ret,
            'vol_ratio': vol_ratio,
            'n_trades': len(self.trades),
        }


# ─────────────────────────────────────────────────────────────
#  机构意图检测器（主类）
# ─────────────────────────────────────────────────────────────

class IntentDetector:
    """
    整合订单簿 + 成交流，检测6种机构意图。

    检测逻辑（严格对应你给的框架）:

    1. BULL_TRAP (诱多):
       - 被动卖单大量堆积 (passive_ask_size > threshold)
       - 突然出现大量撤单 (passive_ask_size 急剧下降)
       → 机构准备拉升后砸盘

    2. BEAR_TRAP (诱空):
       - 被动买单大量堆积 (passive_bid_size > threshold)
       - 突然出现大量撤单 (passive_bid_size 急剧下降)
       → 机构准备砸盘后拉升

    3. ACCUMULATING (暗中吸货):
       - 主动买入量大 (buy_ratio > 0.6)
       - 但价格不跟涨 (ret < 0.1% 或 ret < spread)
       → 大资金在暗中收集

    4. DISTRIBUTING (暗中派发):
       - 主动卖出量大 (buy_ratio < 0.4)
       - 但价格不跟跌 (ret > -0.1% 或 |ret| < spread)
       → 大资金在暗中派发

    5. LIQUIDITY_PROBE (测试流动性):
       - 某价位有大单挂着 (size > threshold)
       - 反复出现但很少成交 (挂成交比低)
       → 机构在诱导/测试流动性

    6. MICRO_DRIFT (方向信号):
       - Micro-Price 持续高于实际价格 (drift > threshold)
       - 或者持续低于实际价格
       → 方向信号
    """

    def __init__(
        self,
        passive_size_thresh: float = 5.0,    # BTC，大单阈值
        passive_count_thresh: int = 3,
        cancel_size_thresh: float = 2.0,     # BTC，触发撤单检测的最小量
        micro_drift_thresh_bps: float = 2.0,  # micro-price偏离超过2bps
        oi_thresh: float = 0.4,              # OI绝对值超过0.4
        vpin_thresh: float = 0.5,             # VPIN超过0.5
        price_lag_thresh_pct: float = 0.05,  # 价格变动 < 0.05% 认为不跟涨/跌
        probe_size_thresh: float = 10.0,     # BTC，探流动性大单
        probe_cooldown_s: float = 30.0,      # 同一价格探流动性冷却时间
    ):
        self.ob_analyzer = OrderBookAnalyzer()
        self.trade_analyzer = TradeFlowAnalyzer()
        self.passive_size_thresh = passive_size_thresh
        self.passive_count_thresh = passive_count_thresh
        self.cancel_size_thresh = cancel_size_thresh
        self.micro_drift_thresh_bps = micro_drift_thresh_bps
        self.oi_thresh = oi_thresh
        self.vpin_thresh = vpin_thresh
        self.price_lag_thresh_pct = price_lag_thresh_pct
        self.probe_size_thresh = probe_size_thresh
        self.probe_cooldown_s = probe_cooldown_s

        # 探流动性冷却
        self._probe_history: dict[float, float] = {}  # price → last_probe_ts

        # 被动单历史（用于检测突然撤单）
        self._passive_ask_history: deque[float] = deque(maxlen=30)
        self._passive_bid_history: deque[float] = deque(maxlen=30)

        # 信号缓冲（用于连续确认）
        self._signal_buffer: deque[IntentSignal] = deque(maxlen=10)

        # 信号冷却计数器
        self._update_counter: int = 0
        self._last_signal_intent: str | None = None
        self._last_signal_update: int = -999
        self._last_signal_ts: float = 0.0   # Unix时间戳
        self._cooldown_s: float = 5.0         # 同类型信号最短间隔（秒）

        self._last_cancel = None

    def update_depth(self, bids_raw, asks_raw, update_id: int):
        self.ob_analyzer.update_from_depth_msg(bids_raw, asks_raw, update_id)
        self._detect_passive_changes()

    def update_trade(self, trade: TradeEvent):
        self.trade_analyzer.add_trade(trade)

    def _detect_passive_changes(self):
        """追踪被动单堆积变化."""
        pp = self.ob_analyzer.get_passive_pressure()
        self._passive_ask_history.append(pp['passive_ask_size'])
        self._passive_bid_history.append(pp['passive_bid_size'])

    def detect(self) -> IntentSignal | None:
        """
        执行全部6种检测，返回第一个命中的信号或None。
        优先级: BULL_TRAP > BEAR_TRAP > ACCUMULATING > DISTRIBUTING > LIQUIDITY_PROBE > MICRO_DRIFT
        """
        self._update_counter += 1

        signals = []

        # ── 1. BULL_TRAP: 被动卖单堆积 + 突然撤单 ─────────────────────────
        s = self._detect_bull_trap()
        if s: signals.append(s)

        # ── 2. BEAR_TRAP: 被动买单堆积 + 突然撤单 ─────────────────────────
        s = self._detect_bear_trap()
        if s: signals.append(s)

        # ── 3. ACCUMULATING: 主动买入 + 价格不跟涨 ─────────────────────────
        s = self._detect_accumulating()
        if s: signals.append(s)

        # ── 4. DISTRIBUTING: 主动卖出 + 价格不跟跌 ─────────────────────────
        s = self._detect_distributing()
        if s: signals.append(s)

        # ── 5. LIQUIDITY_PROBE: 大单挂而不成交 + 反复撤挂 ─────────────────
        s = self._detect_liquidity_probe()
        if s: signals.append(s)

        # ── 6. MICRO_DRIFT: Micro-Price 持续偏离 ───────────────────────────
        s = self._detect_micro_drift()
        if s: signals.append(s)

        if not signals:
            return None

        # 返回置信度最高的信号（带冷却）
        signals.sort(key=lambda x: x.confidence, reverse=True)
        best = signals[0]

        # Cooldown: same intent must wait 5 updates AND 5 seconds
        if (best.intent.value == self._last_signal_intent and
                (self._update_counter - self._last_signal_update < 5 or
                 time.time() - self._last_signal_ts < self._cooldown_s)):
            return None

        self._last_signal_intent = best.intent.value
        self._last_signal_update = self._update_counter
        self._last_signal_ts = time.time()
        return best

    def _detect_bull_trap(self) -> IntentSignal | None:
        """
        被动卖单大量堆积（价格高于卖一价的大卖单）
        + 突然大量撤单
        = 诱多信号（机构准备拉升后砸盘）

        特征: passive_ask_size 突然下降
        """
        pp = self.ob_analyzer.get_passive_pressure()
        current_passive_ask = pp['passive_ask_size']

        if len(self._passive_ask_history) < 5:
            return None

        # 检测被动卖单堆积: 当前 passive_ask_size 明显高于历史平均
        hist_mean = statistics.mean(self._passive_ask_history)
        hist_max = max(self._passive_ask_history)

        if current_passive_ask < hist_mean * 0.7 and hist_max > self.passive_size_thresh:
            # 被动卖单突然减少（大量撤单）
            # 检查是否是从高水平撤下
            delta = hist_max - current_passive_ask
            if delta > self.cancel_size_thresh:
                # 触发诱多信号
                confidence = min(1.0, (delta / self.passive_size_thresh) * 0.8)
                return IntentSignal(
                    intent=Intent.BULL_TRAP,
                    confidence=confidence,
                    price=pp['best_ask'],
                    ts=int(time.time() * 1000),
                    metadata={
                        'passive_ask_before': hist_max,
                        'passive_ask_after': current_passive_ask,
                        'cancel_delta': delta,
                        'type': 'passive_ask_withdrawal',
                    }
                )

        return None

    def _detect_bear_trap(self) -> IntentSignal | None:
        """
        被动买单大量堆积 + 突然撤单
        = 诱空信号（机构准备砸盘后拉升）
        """
        pp = self.ob_analyzer.get_passive_pressure()
        current_passive_bid = pp['passive_bid_size']

        if len(self._passive_bid_history) < 5:
            return None

        hist_mean = statistics.mean(self._passive_bid_history)
        hist_max = max(self._passive_bid_history)

        if current_passive_bid < hist_mean * 0.7 and hist_max > self.passive_size_thresh:
            delta = hist_max - current_passive_bid
            if delta > self.cancel_size_thresh:
                confidence = min(1.0, (delta / self.passive_size_thresh) * 0.8)
                return IntentSignal(
                    intent=Intent.BEAR_TRAP,
                    confidence=confidence,
                    price=pp['best_bid'],
                    ts=int(time.time() * 1000),
                    metadata={
                        'passive_bid_before': hist_max,
                        'passive_bid_after': current_passive_bid,
                        'cancel_delta': delta,
                        'type': 'passive_bid_withdrawal',
                    }
                )
        return None

    def _detect_accumulating(self) -> IntentSignal | None:
        """
        主动买入（taker-buy）大量发生
        + 但价格没有跟随上涨
        = 机构暗中吸货

        逻辑: buy_ratio高 但 price不涨
        """
        metrics = self.trade_analyzer.get_metrics()
        oi = metrics['oi']
        vpin = metrics['vpin']
        buy_ratio = metrics['buy_ratio']
        recent_ret = metrics['recent_ret_pct']
        vol_ratio = metrics['vol_ratio']

        best_bid_price = max(self.ob_analyzer.ob.bids.keys()) if self.ob_analyzer.ob.bids else 0
        best_ask_price = min(self.ob_analyzer.ob.asks.keys()) if self.ob_analyzer.ob.asks else 0
        spread = best_ask_price - best_bid_price

        if not self.ob_analyzer.ob.bids or not self.ob_analyzer.ob.asks:
            return None

        # 条件1: 主动买入占主导 (buy_ratio > 0.6)
        if buy_ratio < 0.6:
            return None

        # 条件2: 价格不跟涨 (ret < 0.05% 或 ret < spread_usd)
        if recent_ret >= self.price_lag_thresh_pct:
            return None

        # 条件3: 高成交量（吸筹通常伴随大量）
        if vol_ratio < 1.5:
            return None

        # 综合置信度
        confidence = min(1.0, (buy_ratio - 0.5) * 1.5 + (1 - recent_ret / 0.1) * 0.3)
        if confidence < 0.50:
            return None

        return IntentSignal(
            intent=Intent.ACCUMULATING,
            confidence=confidence,
            price=best_ask_price,
            ts=int(time.time() * 1000),
            metadata={
                'buy_ratio': round(buy_ratio, 3),
                'oi': round(oi, 3),
                'vpin': round(vpin, 3),
                'recent_ret_pct': round(recent_ret, 4),
                'vol_ratio': round(vol_ratio, 2),
                'spread': round(spread, 2),
                'type': 'buy_pressure_no_price_rise',
            }
        )

    def _detect_distributing(self) -> IntentSignal | None:
        """
        主动卖出（taker-sell）大量发生
        + 但价格没有跟随下跌
        = 机构暗中派发
        """
        metrics = self.trade_analyzer.get_metrics()
        buy_ratio = metrics['buy_ratio']
        recent_ret = metrics['recent_ret_pct']
        vol_ratio = metrics['vol_ratio']

        best_bid = self.ob_analyzer.ob.bids[min(self.ob_analyzer.ob.bids.keys())] if self.ob_analyzer.ob.bids else 0
        best_ask = self.ob_analyzer.ob.asks[max(self.ob_analyzer.ob.asks.keys())] if self.ob_analyzer.ob.asks else 0

        if buy_ratio > 0.4:
            return None

        if recent_ret <= -self.price_lag_thresh_pct:
            return None

        if vol_ratio < 1.5:
            return None

        confidence = min(1.0, (0.5 - buy_ratio) * 1.5 + (recent_ret + 0.1) * 0.3)
        if confidence < 0.50:
            return None

        return IntentSignal(
            intent=Intent.DISTRIBUTING,
            confidence=confidence,
            price=best_bid,
            ts=int(time.time() * 1000),
            metadata={
                'buy_ratio': round(buy_ratio, 3),
                'recent_ret_pct': round(recent_ret, 4),
                'vol_ratio': round(vol_ratio, 2),
                'type': 'sell_pressure_no_price_drop',
            }
        )

    def _detect_liquidity_probe(self) -> IntentSignal | None:
        """
        大单挂在订单簿上但不怎么成交
        + 反复出现（撤了又挂，挂了又撤）

        逻辑: 某价格有大单, 成交率低, 重复出现
        """
        now = time.time()

        # 检查所有超过阈值的大单
        large_asks = self.ob_analyzer.get_large_passive_orders('ask', self.probe_size_thresh)
        large_bids = self.ob_analyzer.get_large_passive_orders('bid', self.probe_size_thresh)

        for price, size in large_asks + large_bids:
            side = 'ask' if price > (self.ob_analyzer.ob.asks[max(self.ob_analyzer.ob.asks.keys())] if self.ob_analyzer.ob.asks else 0) else 'bid'

            # 冷却时间
            if price in self._probe_history:
                if now - self._probe_history[price] < self.probe_cooldown_s:
                    continue

            # 成交率检查：看看这个价格附近有多少成交
            trades_at_price = sum(1 for t in self.trade_analyzer.trades if abs(t.price - price) < price * 0.001)
            fill_rate = trades_at_price / max(len(self.trade_analyzer.trades), 1)

            if size > self.probe_size_thresh and fill_rate < 0.1:
                self._probe_history[price] = now
                confidence = min(1.0, size / self.probe_size_thresh * 0.7)
                return IntentSignal(
                    intent=Intent.LIQUIDITY_PROBE,
                    confidence=confidence,
                    price=price,
                    ts=int(time.time() * 1000),
                    metadata={
                        'side': side,
                        'size': round(size, 4),
                        'price': round(price, 2),
                        'type': 'large_order_low_fill_rate',
                    }
                )

        return None

    def _detect_micro_drift(self) -> IntentSignal | None:
        """
        Micro-Price 持续偏离实际价格
        = 方向信号
        """
        mp = self.ob_analyzer.compute_micro_price()
        best_bid = min(self.ob_analyzer.ob.bids.keys()) if self.ob_analyzer.ob.bids else 0
        best_ask = max(self.ob_analyzer.ob.asks.keys()) if self.ob_analyzer.ob.asks else 0
        mid_price = (best_bid + best_ask) / 2

        if mid_price == 0:
            return None

        drift_bps = abs(mp - mid_price) / mid_price * 10_000

        # 更新历史
        self.ob_analyzer.micro_price_history.append(mp)
        mp_history = list(self.ob_analyzer.micro_price_history)

        if len(mp_history) < 10:
            return None

        # 检测趋势: 持续偏向上或偏向下
        above = sum(1 for v in mp_history if v > mid_price * (1 + 0.0001))
        below = sum(1 for v in mp_history if v < mid_price * (1 - 0.0001))

        if drift_bps < self.micro_drift_thresh_bps:
            return None

        # 持续偏离超过阈值
        if above >= 7 or below >= 7:
            direction = 'upward' if above > below else 'downward'
            confidence = min(1.0, drift_bps / 10 * 0.6 + max(above, below) / len(mp_history) * 0.4)
            if confidence < 0.50:
                return None

            return IntentSignal(
                intent=Intent.MICRO_DRIFT,
                confidence=confidence,
                price=mid_price,
                ts=int(time.time() * 1000),
                metadata={
                    'micro_price': round(mp, 2),
                    'mid_price': round(mid_price, 2),
                    'drift_bps': round(drift_bps, 2),
                    'direction': direction,
                    'above_count': above,
                    'below_count': below,
                }
            )

        return None

    def get_state(self) -> dict:
        """返回当前状态快照（用于调试/监控）."""
        pp = self.ob_analyzer.get_passive_pressure()
        metrics = self.trade_analyzer.get_metrics()
        mp = self.ob_analyzer.compute_micro_price()
        mid = (pp['best_bid'] + pp['best_ask']) / 2

        return {
            'passive_bid_size': round(pp['passive_bid_size'], 4),
            'passive_ask_size': round(pp['passive_ask_size'], 4),
            'spread': round(pp['spread'], 2),
            'buy_ratio': round(metrics['buy_ratio'], 3),
            'oi': round(metrics['oi'], 3),
            'vpin': round(metrics['vpin'], 3),
            'vol_ratio': round(metrics['vol_ratio'], 2),
            'micro_price': round(mp, 2),
            'mid_price': round(mid, 2),
            'micro_drift_bps': round(abs(mp - mid) / mid * 10_000, 3) if mid else 0,
            'n_trades': metrics['n_trades'],
        }


# ─────────────────────────────────────────────────────────────
#  Binance WebSocket 连接管理器
# ─────────────────────────────────────────────────────────────

class BinanceWebSocket:
    """
    连接 Binance WebSocket:
    - <symbol>@aggTrade    → 逐笔成交
    - <symbol>@depth20@100ms → 订单簿20档
    """

    def __init__(self, symbol: str, on_trade, on_depth, on_signal):
        self.symbol = symbol.lower()
        self.on_trade = on_trade
        self.on_depth = on_depth
        self.on_signal = on_signal
        self.detector = IntentDetector()
        self._ws = None
        self._session = None
        self._running = False

    async def start(self):
        self._session = aiohttp.ClientSession()
        self._running = True

        streams = [
            f"{self.symbol}@aggTrade",
            f"{self.symbol}@depth20@100ms",
        ]
        ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        async with self._session.ws_connect(ws_url) as ws:
            self._ws = ws
            async for msg in ws:
                if not self._running:
                    break
                data = json.loads(msg.data)
                stream = data.get('stream', '')
                payload = data.get('data', {})

                if 'aggTrade' in stream:
                    self._handle_trade(payload)
                elif 'depth' in stream:
                    self._handle_depth(payload)

    def _handle_trade(self, payload: dict):
        """处理 aggTrade 消息."""
        # Binance aggTrade fields:
        # {
        #   "e": "aggTrade",      // event type
        #   "s": "BTCUSDT",       // symbol
        #   "p": "73352.01",      // price
        #   "q": "0.710",         // quantity
        #   "m": true,            // is buyer maker (True= seller was taker)
        #   "T": 1748454000000,   // trade time
        #   "a": 12345            // trade id
        # }
        trade = TradeEvent(
            price=float(payload['p']),
            size=float(payload['q']),
            is_buyer_maker=payload['m'],
            side='buy' if not payload['m'] else 'sell',  # not m = buyer was taker
            ts=int(payload['T']),
            trade_id=int(payload['a']),
        )
        self.detector.update_trade(trade)
        if self.on_trade is not None:
            self.on_trade(trade)

    def _handle_depth(self, payload: dict):
        """处理 depth 消息 (Binance @depth20@100ms)."""
        bids_raw = payload.get('bids', payload.get('b', []))
        asks_raw = payload.get('asks', payload.get('a', []))
        update_id = int(payload.get('u', payload.get('lastUpdateId', 0)))

        self.detector.update_depth(bids_raw, asks_raw, update_id)
        if self.on_depth is not None:
            self.on_depth(bids_raw, asks_raw)

        # 检测信号
        signal = self.detector.detect()
        if signal:
            self.on_signal(signal)

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


# ─────────────────────────────────────────────────────────────
#  运行器
# ─────────────────────────────────────────────────────────────

async def run_live(symbol: str = 'btcusdt', duration_s: int = 60):
    """实时运行检测器 N 秒，打印信号和状态."""
    print(f"Starting live detection for {symbol.upper()} for {duration_s}s...")
    print(f"  WebSocket: {symbol}@aggTrade + {symbol}@depth20@100ms")
    print("  Thresholds: passive_size=5BTC, cancel=2BTC, micro_drift=2bps, vol_ratio=1.5x")
    print()

    signals_received = []
    last_state_print = 0

    def on_trade(trade: TradeEvent):
        pass  # 不想每笔都打印

    def on_depth(bids, asks):
        nonlocal last_state_print
        now = time.time()
        if now - last_state_print >= 3.0:
            state = detector.get_state()
            print(f"[{time.strftime('%H:%M:%S')}]"
                  f"  bid_vol={state['buy_ratio']:.3f} OI={state['oi']:+.2f}"
                  f"  VPIN={state['vpin']:.3f}  vol_x={state['vol_ratio']:.1f}"
                  f"  p_ask={state['passive_ask_size']:.2f} p_bid={state['passive_bid_size']:.2f}"
                  f"  mp={state['micro_price']} mid={state['mid_price']}"
                  f"  drift={state['micro_drift_bps']:.2f}bps"
                  )
            last_state_print = now

    def on_signal(signal: IntentSignal):
        ts_str = time.strftime('%H:%M:%S', time.localtime(signal.ts / 1000))
        m = signal.metadata
        print(f"\n*** {ts_str}  SIGNAL: {signal.intent.value.upper()}  conf={signal.confidence:.0%}  price={signal.price} ***")
        for k, v in m.items():
            print(f"    {k}: {v}")
        print()
        signals_received.append(signal)

    detector = IntentDetector()
    ws = BinanceWebSocket(symbol, lambda t: None, lambda b, a: None, lambda s: None)
    ws.detector = detector
    last_state_print = 0

    def on_trade(trade: TradeEvent):
        pass  # noisy, skip per-trade printing

    def on_depth(bids, asks):
        nonlocal last_state_print
        now = time.time()
        state = ws.detector.get_state()
        if now - last_state_print >= 3.0:
            print(f"[{time.strftime('%H:%M:%S')}]"
                  f"  buy_ratio={state['buy_ratio']:.3f} OI={state['oi']:+.2f}"
                  f"  VPIN={state['vpin']:.3f}  vol_x={state['vol_ratio']:.1f}"
                  f"  p_ask={state['passive_ask_size']:.3f} p_bid={state['passive_bid_size']:.3f}"
                  f"  spread={state['spread']:.2f}  mp={state['micro_price']}"
                  f"  mid={state['mid_price']}  drift={state['micro_drift_bps']:.2f}bps"
                  f"  n={state['n_trades']}"
                  )
            last_state_print = now

    def on_signal(signal: IntentSignal):
        ts_str = time.strftime('%H:%M:%S', time.localtime(signal.ts / 1000))
        m = signal.metadata
        print(f"\n*** {ts_str}  {signal.intent.value.upper():16s}  conf={signal.confidence:.0%}  price={signal.price}")
        for k, v in m.items():
            print(f"    {k}: {v}")
        signals_received.append(signal)

    ws.on_trade = on_trade
    ws.on_depth = on_depth
    ws.on_signal = on_signal

    try:
        await asyncio.wait_for(ws.start(), timeout=duration_s)
    except asyncio.TimeoutError:
        await ws.stop()

    print(f"\n--- Session complete ---")
    print(f"  Total signals: {len(signals_received)}")
    for s in signals_received:
        print(f"  - {s.intent.value}: conf={s.confidence:.0%} @ price={s.price}")
    if not signals_received:
        print("  No signals triggered (normal for calm market)")


# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Live institutional intent detector')
    parser.add_argument('--symbol', default='btcusdt')
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()

    asyncio.run(run_live(args.symbol, args.duration))