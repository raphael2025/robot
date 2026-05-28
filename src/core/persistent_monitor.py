"""
persistent_monitor.py
======================
持续运行：监听Binance WebSocket，积累信号→追踪价格→定期输出胜率报告。

用法: python persistent_monitor.py --symbol btcusdt --interval 600
"""
import asyncio, json, sys, time, os
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, '/home/raphael/robot/src')
sys.path.insert(0, '/home/raphael/robot/src/core')

from core.orderflow_detector import IntentDetector, Intent, BinanceWebSocket

RESULTS_DIR = '/home/raphael/robot/results/monitor'
os.makedirs(RESULTS_DIR, exist_ok=True)
STATE_FILE = os.path.join(RESULTS_DIR, 'signal_log.json')


def fetch_current_price(symbol: str) -> float:
    import urllib.request, json
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return float(json.loads(r.read())['price'])
    except Exception:
        return None


async def main(symbol: str, interval: int = 600):
    detector = IntentDetector()
    signal_log = []
    pending = []  # shared reference, passed to tracker

    def on_signal(sig):
        now_str = datetime.now().strftime('%H:%M:%S')
        trigger_price = detector.ob_analyzer.micro_price or fetch_current_price(symbol) or 0
        item = {
            'time': now_str, 'sig_ts': time.time(),
            'intent': sig.intent.value, 'conf': sig.confidence,
            'trigger_price': trigger_price,
            '1m': None, '3m': None, '5m': None, '15m': None,
        }
        signal_log.append(item)
        pending.append(item)
        print(f"[{now_str}] {sig.intent.value:<20} conf={sig.confidence:.0%}  price={trigger_price:.2f}")
        _save_log(signal_log)

    ws = BinanceWebSocket(symbol, on_trade=None, on_depth=None, on_signal=on_signal)

    # Price tracker (independent loop)
    async def _track_prices(pending, symbol):
        while True:
            cp = fetch_current_price(symbol)
            now = time.time()
            to_remove = []
            for item in pending:
                for key, secs in [('1m', 60), ('3m', 180), ('5m', 300), ('15m', 900)]:
                    if item.get(key) is None and (now - item['sig_ts']) >= secs:
                        item[key] = (cp - item['trigger_price']) / item['trigger_price'] * 10000
                        break
                if item.get('15m') is not None:
                    to_remove.append(item)
            for item in to_remove:
                pending.remove(item)
            await asyncio.sleep(15)

    tracker = asyncio.create_task(_track_prices(pending, symbol))
    await ws.start()

    # Report loop
    last_report = time.time()
    while True:
        await asyncio.sleep(interval)
        stats = _compute_stats(signal_log)
        elapsed = time.time() - last_report
        report = _format_report(stats, len(signal_log), elapsed)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        for fname in [f'{RESULTS_DIR}/report_{ts}.txt', f'{RESULTS_DIR}/latest.txt']:
            with open(fname, 'w') as f: f.write(report)
        print('\n' + report)
        _save_log(signal_log)
        last_report = time.time()


def _save_log(log):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump([{k: (v if v is not None else None) for k, v in item.items()} for item in log], f, indent=2)
    except Exception:
        pass


def _compute_stats(log):
    by_intent = defaultdict(list)
    for item in log:
        if item.get('1m') is not None:
            by_intent[item['intent']].append(item)

    expect = {'ACCUMULATING': 1, 'BEAR_TRAP': 1, 'DISTRIBUTING': -1, 'BULL_TRAP': -1}
    out = {}
    for intent, items in by_intent.items():
        ex = expect.get(intent, 0)
        def wr(rets):
            if not rets or ex == 0: return 0.0, []
            return sum(1 for r in rets if r * ex > 0) / len(rets), list(rets)
        rows = {'n': len(items)}
        for key in ['1m', '3m', '5m', '15m']:
            rets = [item[key] for item in items if item.get(key) is not None]
            w, rs = wr(rets)
            rows[f'{key}_wr'] = w
            rows[f'{key}_avg'] = sum(rs) / len(rs) if rs else 0.0
            rows[f'{key}_n'] = len(rs)
        out[intent] = rows
    return out


def _format_report(stats, total, elapsed):
    h = elapsed / 3600
    lines = [
        f"═══════════════════════════════════════════════",
        f"  机构意图检测持续监控报告",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  运行: {h:.1f}h  信号: {total}",
        f"{'─'*55}",
        f"{'Intent':<18} {'N':>4}  {'1m WR':>9} {'Avg':>10}  {'3m WR':>9} {'Avg':>10}  {'5m WR':>9} {'Avg':>10}  {'15m WR':>9} {'Avg':>10}",
        f"{'─'*110}",
    ]
    for intent in ['ACCUMULATING', 'DISTRIBUTING', 'BULL_TRAP', 'BEAR_TRAP', 'LIQUIDITY_PROBE', 'MICRO_DRIFT']:
        if intent not in stats: continue
        s = stats[intent]
        def cell(wr, avg, n):
            return f"{wr:.0%}  {avg:>+8.1f}  ({n})" if n else "   --        --      "
        lines.append(
            f"{intent:<18} {s['n']:>4}  "
            f"{cell(s['1m_wr'],s['1m_avg'],s['1m_n'])}  "
            f"{cell(s['3m_wr'],s['3m_avg'],s['3m_n'])}  "
            f"{cell(s['5m_wr'],s['5m_avg'],s['5m_n'])}  "
            f"{cell(s['15m_wr'],s['15m_avg'],s['15m_n'])}"
        )
    lines += ["", "判断: ACCUMULATING/BEAR_TRAP→涨预期, BULL_TRAP/DISTRIBUTING→跌预期"]
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='btcusdt')
    p.add_argument('--interval', type=int, default=600)
    args = p.parse_args()
    print(f"Starting: {args.symbol}, report every {args.interval}s")
    asyncio.run(main(args.symbol, args.interval))