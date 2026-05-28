"""
collector.py
===========
Fetch and persist Binance 1m klines to local Parquet files.
Build up to --days of history, then save to disk for use by
backtest_engine.py without needing network.
"""
import sys, time, requests, argparse
import pandas as pd

argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="BTCUSDT")
parser.add_argument("--days", type=int, default=30)
parser.add_argument("--interval", default="1m")
parser.add_argument("--limit", type=int, default=1000)
args = parser.parse_args()

ms_per_day = 86_400_000
start_ms = int(time.time() * 1000) - args.days * ms_per_day
end_ms = int(time.time() * 1000)

all_klines = []
chunk_ms = args.days * ms_per_day
chunk_days = 30
chunks = (args.days + chunk_days - 1) // chunk_days

url = "https://api.binance.com/api/v3/klines"
params = {"symbol": args.symbol, "interval": args.interval, "limit": args.limit}

print(f"Fetching up to {args.days}d of {args.interval} klines for {args.symbol}...")
fetched = 0
while start_ms < end_ms:
    p = dict(params, startTime=start_ms, endTime=min(start_ms + chunk_ms, end_ms))
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        break
    all_klines.extend(rows)
    fetched += len(rows)
    start_ms = rows[-1][0] + 1
    print(f"  {len(rows)} rows  (total: {fetched})", flush=True)
    time.sleep(0.2)

print(f"Total rows: {len(all_klines)}")

# Parse
cols = ["open_time","open","high","low","close","vol","close_time","qav","trades","tbb","tbq","ignore"]
df = pd.DataFrame(all_klines, columns=cols)
for c in ["open","high","low","close","vol","qav","tbb","tbq"]:
    df[c] = pd.to_numeric(df[c])
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
df = df.rename(columns={
    "open_time": "ts", "trades": "n_trades",
    "qav": "qvolume", "tbb": "taker_buy_base", "tbq": "taker_buy_quote"
})
df["buy_ratio"] = df["taker_buy_base"] / df["vol"]
df = df.sort_values("ts").reset_index(drop=True)

path = f"/home/raphael/robot/data/{args.symbol}_{args.interval}_{args.days}d.parquet"
pd.read_parquet.__module__  # just to trigger pyarrow import
import os; os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_parquet(path)
print(f"Saved {len(df)} rows to {path}")
print(f"Date range: {df['ts'].min()} → {df['ts'].max()}")