"""
signal_validator.py
====================
追踪实盘信号触发后的价格走向，验证框架是否准确。

对每个触发的信号，记录触发价格 + 之后1/3/5分钟的价格变化。
最后汇总：每个信号类型的胜率、平均盈亏。
"""
import asyncio
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/home/raphael/robot/src')
sys.path.insert(0, '/home/raphael/robot')

from core.orderflow_detector import IntentDetector, Intent
from adapters.exchange import BinanceAdapter


class SignalValidator:
    def __init__(self, symbol: str, poll_interval: int = 5, warmup_seconds: int = 60):
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.warmup = warmup_seconds
        self.detector = IntentDetector()
        self.binance = BinanceAdapter(symbol)
        self.signals: list[dict] = []
        self.price_history: list[dict] = []  # {'ts': unix, 'price': mid}
        self.running = False

    async def _price_recorder(self):
        """后台持续记录价格快照（每5秒）"""
        while self.running:
            try:
                ob = await self.binance.fetch_orderbook_snapshot(limit=20)
                if ob['bids'] and ob['asks']:
                    mid = (float(ob['bids'][0][0]) + float(ob['asks'][0][0])) / 2
                    self.price_history.append({'ts': time.time(), 'price': mid})
            except Exception:
                pass
            await asyncio.sleep(self.poll_interval)

    def _get_price_at(self, ts: float) -> float | None:
        """找到 >= ts 的第一个价格"""
        for p in self.price_history:
            if p['ts'] >= ts:
                return p['price']
        return None

    async def run(self, duration_s: int = 300):
        self.running = True
        recorder = asyncio.create_task(self._price_recorder())
        await asyncio.sleep(self.warmup)  # 先预热积累数据

        print(f"Signal tracking started — {duration_s}s window, checking 1m/3m/5m outcomes...")
        start_ts = time.time()
        last_print = 0

        while time.time() - start_ts < duration_s:
            try:
                trades = await self.binance.fetch_recent_trades(limit=50)
                depth = await self.binance.fetch_orderbook_snapshot(limit=20)

                for t in trades:
                    self.detector.update_trade({
                        'price': float(t['price']),
                        'size': float(t['qty']),
                        'is_buyer_maker': t.get('is_buyer_maker', False),
                        'ts': int(t['time']),
                    })

                if depth['bids'] and depth['asks']:
                    bids = [[b[0], b[1]] for b in depth['bids']]
                    asks = [[a[0], a[1]] for a in depth['asks']]
                    self.detector.update_depth(bids, asks, update_id=depth.get('lastUpdateId', 0))

                sig = self.detector.detect()
                if sig:
                    trigger_price = sig.price
                    now = time.time()
                    self.signals.append({
                        'ts': now,
                        'intent': sig.intent.value,
                        'confidence': sig.confidence,
                        'trigger_price': trigger_price,
                        'metadata': sig.metadata,
                    })
                    print(f"  SIGNAL {sig.intent.value:15s} @ {trigger_price:.2f}  conf={sig.confidence:.0%}")

                if time.time() - last_print > 30:
                    print(f"  [{int(time.time()-start_ts)}s] tracked {len(self.signals)} signals, {len(self.price_history)} price samples")
                    last_print = time.time()

            except Exception as e:
                print(f"  error: {e}")

            await asyncio.sleep(self.poll_interval)

        self.running = False
        recorder.cancel()
        await asyncio.sleep(0.5)
        self._report()

    def _report(self):
        print("\n" + "=" * 70)
        print("SIGNAL VALIDATION REPORT")
        print("=" * 70)

        if not self.signals:
            print("No signals triggered.")
            return

        outcomes = defaultdict(lambda: {'1m': [], '3m': [], '5m': []})
        confirm = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0, 'inconclusive': 0})

        for sig in self.signals:
            intent = sig['intent']
            trigger_ts = sig['ts']
            trigger_price = sig['trigger_price']

            for label, delta in [('1m', 60), ('3m', 180), ('5m', 300)]:
                future_ts = trigger_ts + delta
                future_price = self._get_price_at(future_ts)
                if future_price:
                    pct = (future_price - trigger_price) / trigger_price * 100
                    outcomes[intent][label].append(pct)

            # 判断正确性
            confirm[intent]['total'] += 1
            p1m = outcomes[intent]['1m']
            if p1m:
                last_ret = p1m[-1]  # 1分钟后价格变化

                if intent == 'accumulating':
                    # 主动买入+价格不跟涨 → 应该涨
                    correct = last_ret > 0.001
                elif intent == 'distributing':
                    # 主动卖出+价格不跟跌 → 应该跌
                    correct = last_ret < -0.001
                elif intent == 'bull_trap':
                    # 被动卖单撤 → 准备砸盘（看空）
                    correct = last_ret < -0.001
                elif intent == 'bear_trap':
                    # 被动买单撤 → 准备拉升（看多）
                    correct = last_ret > 0.001
                elif intent == 'liquidity_probe':
                    # 大单挂而不成交 → 方向不明，看是否收复
                    correct = abs(last_ret) < 0.01  # 窄幅横盘=诱导成功
                elif intent == 'micro_drift':
                    # micro_price偏 → 方向跟随micro_price
                    mp = sig['metadata'].get('micro_price', trigger_price)
                    correct = (future_price - trigger_price) * (mp - trigger_price) > 0
                else:
                    correct = None

                if correct is True:
                    confirm[intent]['correct'] += 1
                elif correct is False:
                    confirm[intent]['wrong'] += 1
                else:
                    confirm[intent]['inconclusive'] += 1

        print(f"\n{'Signal':<20} {'N':>4}  {'1m avg%':>10}  {'3m avg%':>10}  {'5m avg%':>10}  {'WinRate':>10}")
        print("-" * 70)
        for intent in ['accumulating', 'distributing', 'bull_trap', 'bear_trap', 'liquidity_probe', 'micro_drift']:
            data = outcomes[intent]
            n = len(data['1m'])
            if n == 0:
                continue
            avg1 = sum(data['1m']) / n
            avg3 = sum(data['3m']) / n if data['3m'] else 0
            avg5 = sum(data['5m']) / n if data['5m'] else 0
            c = confirm[intent]
            wr = c['correct'] / max(c['total'], 1)
            print(f"{intent:<20} {n:>4}  {avg1:>+10.4f}  {avg3:>+10.4f}  {avg5:>+10.4f}  {wr:>10.1%}")

        print()
        for intent, c in confirm.items():
            total = c['total']
            if total == 0:
                continue
            wr = c['correct'] / total
            print(f"  {intent}: {c['correct']}/{total} correct ({wr:.0%})")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='btcusdt')
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--poll', type=int, default=5)
    args = parser.parse_args()

    v = SignalValidator(args.symbol, poll_interval=args.poll, warmup_seconds=60)
    await v.run(duration_s=args.duration)


if __name__ == '__main__':
    asyncio.run(main())