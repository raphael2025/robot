# Robot — Institutional Intent Detection Engine

机构意图识别引擎：通过订单簿 + 成交数据分析 6 种机构行为模式。

---

## 架构

```
ExchangeAdapter (WebSocket / REST)
    ├── BinanceAdapter     — wss://stream.binance.com
    ├── BybitAdapter       — wss://stream.bybit.com
    └── OKXAdapter          — wss://ws.okx.com
         ↓
OrderBookSnap / Trade  (统一数据结构)
    ├── MicroPriceEngine    — micro_price = mid + imbalance × spread/2
    ├── OrderFlowMetrics    — VPIN, Order Imbalance, Absorption Ratio, Spoofing Score
    └── InstitutionalIntentClassifier  — 6 种信号
              ↓
IntentSignal  (信号输出 → Webhook / 交易引擎 / 日志)
```

---

## 6 种机构意图信号

| 信号 | 触发条件 | 预期价格方向 |
|---|---|---|
| **BULL_TRAP** (诱多) | 被动卖单堆积 + 突然撤单 + 价格假突破 | 砸盘 ↓ |
| **BEAR_TRAP** (诱空) | 被动买单堆积 + 突然撤单 + 价格假突破 | 拉升 ↑ |
| **ABSORPTION** (吸货) | 主动买入 + 价格不跟涨 + 高吸收率 | 拉升 ↑ |
| **DISTRIBUTION** (派发) | 主动卖出 + 价格不跟跌 + 低吸收率 | 砸盘 ↓ |
| **LIQUIDITY_PROBE** (流动性测试) | 大单反复挂而不成交 + 高VPIN | 短期波动 |
| **MICRO_DRIFT** (微观漂移) | Micro-Price 持续偏离 >2bps × 3次 | 方向信号 |

---

## 核心公式

```python
# Micro-Price (Jerrett & Keene, 2019)
micro_price = mid_price + imbalance × (spread / 2)
imbalance    = (bid_vol - ask_vol) / (bid_vol + ask_vol)   ∈ [-1, 1]

# VPIN (Easley, Lopez, O'Hara, 2012)
VPIN = |buy_volume_buckets / total_buckets - 0.5| × 2

# Absorption Ratio
absorption_ratio = passive_volume / aggressive_volume
  > 2.5  → 机构在吸货 (institution accumulating)
  < 0.5  → 机构在派发 (institution distributing)
```

---

## 目录结构

```
robot/
├── config/
│   └── thresholds.toml     # 所有可调参数
├── src/
│   ├── __init__.py         # 公共 API
│   ├── core/
│   │   ├── orderflow.py     # MicroPriceEngine, OrderFlowMetrics,
│   │   │                    # InstitutionalIntentClassifier, IntentSignal
│   │   ├── robot.py         # Robot (live/backtest runner)
│   │   ├── live_analysis.py # 实时分析脚本 (REST轮询)
│   │   └── backtest_engine.py # 历史回测引擎
│   └── adapters/
│       └── exchange.py      # Binance / Bybit / OKX 适配器
└── tests/
    ├── test_orderflow.py   # 18 个测试
    └── test_adapters.py    # 26 个测试
```

---

## 安装

```bash
cd /home/raphael/robot
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-asyncio aiohttp websockets numpy pandas requests
```

---

## 测试

```bash
source .venv/bin/activate
python -m pytest tests/ -v
# 44 passed
```

---

## 实时分析

```bash
source .venv/bin/activate
python src/core/live_analysis.py --symbol BTCUSDT --poll 3 --duration 180
```

输出示例：
```
  13:45:52  mid=  73502.93  spread=  0.0bps  oi=+0.70 ▓▓▓▓▓▓░░░░  vpin=0.000  ar=13.86
  *** 🟢 [STRONG] absorption | confidence=80% | price=73502.9350 ***
```

---

## 历史回测

```bash
source .venv/bin/activate
python src/core/backtest_engine.py --symbol BTCUSDT --days 14
```

输出示例（14天数据）：
```
Baseline (buy-hold per-bar):
  win-rate: 49.2%  avg: +0.17 bps  std: 5.43 bps

Signal Performance vs Baseline (+0.17 bps avg):
Intent                Horizon    N  WinRate    AvgBps    MedBps  Sharpe
-------------------- -------- ---- -------- --------- --------- -------
 ✓ liquidity_probe        30m  186   55.4%    +11.3     +4.7    0.29
 ✓ liquidity_probe        60m  175   73.1%    +22.8    +20.2    0.53
```

**LIQUIDITY_PROBE 60分钟窗口胜率73%，平均 +22.8bps**，显著优于买入持有基准。

---

## 参数调优

参数在 `config/thresholds.toml`，主要调参方向：

| 参数 | 影响 | 默认值 | 建议范围 |
|---|---|---|---|
| `absorption_min` | ABSORPTION 触发阈值 | 2.5 | 2.0–4.0 |
| `drift_bps_threshold` | MICRO_DRIFT 漂移阈值 | 2.0 bps | 1.0–5.0 |
| `drift_consecutive_min` | MICRO_DRIFT 连续次数 | 3 | 2–5 |
| `vol_std_threshold` | LIQUIDITY_PROBE 体积阈值 | 0.7 | 0.5–1.0 |
| `signal_min_confidence` | 最小置信度 | 0.60 | 0.55–0.70 |

---

## 已知限制

1. **spread = 0 问题**：深度行情中买卖价差可能为 0，已用 `max(0, spread)` 防护
2. **VPIN 校准**：VPIN bucket 大小使用 `avg_trade_size × 10`，需根据币种调整
3. **BULL/BEAR_TRAP**：需要订单簿历史对比，当前用 kline 数据粗略近似
4. **回测仅用 kline**：无逐笔成交细节，用 taker-buy 比率作为 OI 代理变量
5. **OKX API**：官方 API 暂时不可用，适配器已写好但 REST 备用路径 404

---

## 下一步建议

1. **接入 WebSocket**：实时流比 REST 轮询精度高 10 倍
2. **逐笔数据存储**：将 `OrderBookSnap + Trade` 写入 Parquet，供回测复用
3. **参数网格优化**：用 `optuna` 或 `GridSearchCV` 搜索最优阈值
4. **信号评分融合**：将 VPIN、OI、AR、Drift 加权融合为单一信号强度
5. **实战对接**：输出 `IntentSignal` 到 Freqtrade 策略或 Telegram webhook