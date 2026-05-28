"""
signal_validator.py — 简化版，直接用requests轮询Binance REST API
"""
import asyncio, json, time, sys, os, urllib.request
from collections import defaultdict

sys.path.insert(0, '/home/raphael/robot/src')
sys.path.insert(0, '/home/raphael/robot')

from core.orderflow_detector import IntentDetector, Intent, TradeEvent


def fetch_binance_ob(symbol: str) -> dict | None:
    try:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit=20"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return None


def fetch_binance_trades(symbol: str) -> list:
    try:
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol.upper()}&limit=100"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return []


async def run(symbol: str, duration_s: int, poll_s: int):
    warmup = 30
    results = {'symbol': symbol, 'start_ts': time.time(), 'signals': [], 'price_log': []}
    detector = IntentDetector()

    print(f"[0s] Warming up {warmup}s...")
    await asyncio.sleep(warmup)

    print(f"[{warmup}s] Validation start — {duration_s}s, poll every {poll_s}s")
    start = time.time()
    last_report = start

    while time.time() - start < duration_s:
        loop = time.time()

        ob = fetch_binance_ob(symbol)
        trades_raw = fetch_binance_trades(symbol)

        for i, t in enumerate(trades_raw):
            # isBuyerMaker=True → seller was taker → aggressive sell → side="sell"
            # isBuyerMaker=False → buyer was taker → aggressive buy → side="buy"
            side = "sell" if t['isBuyerMaker'] else "buy"
            detector.update_trade(TradeEvent(
                price=float(t['price']),
                size=float(t['qty']),
                side=side,
                is_buyer_maker=t['isBuyerMaker'],
                ts=t['time'],
                trade_id=t.get('id', i),
            ))

        if ob and ob['bids'] and ob['asks']:
            bids = [[b[0], b[1]] for b in ob['bids']]
            asks = [[a[0], a[1]] for a in ob['asks']]
            detector.update_depth(bids, asks, update_id=ob['lastUpdateId'])

            mid = (float(ob['bids'][0][0]) + float(ob['asks'][0][0])) / 2
            results['price_log'].append({'ts': time.time(), 'mid': mid})

        sig = detector.detect()
        now = time.time()
        if sig:
            results['signals'].append({
                'ts': now, 'intent': sig.intent.value,
                'confidence': sig.confidence,
                'trigger_mid': mid if ob else sig.price,
                'metadata': sig.metadata,
            })
            print(f"  SIGNAL {sig.intent.value:18s} conf={sig.confidence:.0%} mid={mid:.2f}")

        if now - last_report >= 60:
            print(f"  [{int(now-start)}s] signals={len(results['signals'])} prices={len(results['price_log'])}")
            last_report = now

        elapsed = time.time() - loop
        await asyncio.sleep(max(poll_s - elapsed, 1))

    results['end_ts'] = time.time()

    # ANALYSIS
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    outcomes = defaultdict(lambda: {'1m': [], '3m': [], '5m': []})
    verdict = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'pending': 0})

    def price_after(ts, delta):
        for p in results['price_log']:
            if p['ts'] >= ts + delta:
                return p['mid']
        return None

    for sig in results['signals']:
        intent = sig['intent']
        trigger = sig['trigger_mid']
        sig_ts = sig['ts']

        for label, delta in [('1m', 60), ('3m', 180), ('5m', 300)]:
            fp = price_after(sig_ts, delta)
            if fp:
                outcomes[intent][label].append((fp - trigger) / trigger * 100)

        p1 = price_after(sig_ts, 60)
        if p1 is None:
            verdict[intent]['pending'] += 1
            continue

        ret1 = (p1 - trigger) / trigger * 100
        correct = None
        if intent == 'accumulating':
            correct = ret1 > 0.01
        elif intent == 'distributing':
            correct = ret1 < -0.01
        elif intent == 'bull_trap':
            correct = ret1 < -0.01
        elif intent == 'bear_trap':
            correct = ret1 > 0.01
        elif intent == 'liquidity_probe':
            correct = abs(ret1) < 0.05
        elif intent == 'micro_drift':
            direction = sig['metadata'].get('direction', 'neutral')
            correct = (ret1 > 0) if direction == 'bid_heavy' else (ret1 < 0) if direction == 'ask_heavy' else None

        if correct is True:
            verdict[intent]['correct'] += 1
        elif correct is False:
            verdict[intent]['wrong'] += 1

    elapsed = results['end_ts'] - results['start_ts']
    print(f"\nDuration: {elapsed:.0f}s  Signals: {len(results['signals'])}  Prices: {len(results['price_log'])}\n")

    print(f"{'Intent':<20} {'N':>4}  {'1m avg%':>10}  {'3m avg%':>10}  {'5m avg%':>10}  {'WinRate':>10}")
    print("-" * 72)
    for intent in ['accumulating','distributing','bull_trap','bear_trap','liquidity_probe','micro_drift']:
        d1, d3, d5 = outcomes[intent]['1m'], outcomes[intent]['3m'], outcomes[intent]['5m']
        n = len(d1)
        if n == 0: continue
        avg1 = sum(d1)/n
        avg3 = sum(d3)/n if d3 else float('nan')
        avg5 = sum(d5)/n if d5 else float('nan')
        v = verdict[intent]
        total_v = v['correct'] + v['wrong']
        wr = v['correct'] / total_v if total_v > 0 else float('nan')
        print(f"{intent:<20} {n:>4}  {avg1:>+10.4f}  {avg3:>+10.4f}  {avg5:>+10.4f}  {wr:>10.1%}")

    print()
    for intent, v in verdict.items():
        total = v['correct'] + v['wrong'] + v['pending']
        if total == 0: continue
        wr = v['correct'] / max(v['correct'] + v['wrong'], 1)
        print(f"  {intent}: {v['correct']}/{v['correct']+v['wrong']} correct ({wr:.0%})  pending={v['pending']}")

    os.makedirs('/home/raphael/robot/results', exist_ok=True)
    ts = int(results['start_ts'])
    out = f"/home/raphael/robot/results/{symbol}_validation_{ts}.txt"
    with open(out, 'w') as f:
        f.write(f"Symbol: {symbol}\nStart: {time.ctime(results['start_ts'])}\nDuration: {elapsed:.0f}s\nSignals: {len(results['signals'])}\n\n")
        for sig in results['signals']:
            f.write(f"[{time.ctime(sig['ts'])}] {sig['intent']:20s} conf={sig['confidence']:.0%} mid={sig['trigger_mid']:.2f}\n")
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='btcusdt')
    p.add_argument('--duration', type=int, default=300)
    p.add_argument('--poll', type=int, default=5)
    args = p.parse_args()
    asyncio.run(run(args.symbol, args.duration, args.poll))