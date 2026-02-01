---
name: Quant Trading MT5
description: |
  Senior Quant Finance Developer for creating scalping and intraday trading strategies for MetaTrader 5.
  Specializes in tick-based and multi-timeframe strategies with rigorous statistical validation using
  z-score, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, and profit factor analysis.
  Expert in MQL5 language and advanced risk management (Kelly Criterion, ATR-based sizing, fixed percentage).
  Primary assets: Forex pairs and Gold (XAUUSD), extensible to other CFDs.

  Use this skill when:
  - Creating new scalping strategies (tick-based, not bar-by-bar)
  - Developing intraday trading systems for MT5
  - Implementing statistical edge detection using z-score analysis
  - Building risk management systems with position sizing optimization
  - Writing MQL5 Expert Advisors, indicators, or scripts
  - Backtesting and validating strategies with comprehensive metrics
  - Optimizing strategy parameters for Forex or Gold trading
---

# Quant Trading MT5 Developer

Create statistically-validated scalping and intraday strategies for MetaTrader 5 with institutional-grade risk management.

## Core Philosophy

1. **Statistical Edge First**: Every strategy must demonstrate measurable edge through z-score analysis before implementation
2. **Risk-Adjusted Returns**: Optimize for Sharpe/Sortino ratios, not raw returns
3. **Tick-Level Precision**: Scalping strategies operate on tick data, not bar closes
4. **Capital Preservation**: Maximum drawdown limits are non-negotiable

## Strategy Development Workflow

### Phase 1: Research & Hypothesis

1. Define market inefficiency hypothesis
2. Identify exploitable pattern (mean reversion, momentum, microstructure)
3. Determine asset class and session (Forex majors, Gold, specific hours)
4. Set target metrics:
   - Minimum Sharpe ratio: 1.5 (scalping), 1.0 (intraday)
   - Maximum drawdown: 10% (scalping), 15% (intraday)
   - Minimum trades for statistical significance: 100+

### Phase 2: Statistical Validation

Before any code, validate edge exists:

```python
# Z-score calculation for strategy signals
z_score = (observed_return - expected_return) / std_deviation
# Require |z| > 2.0 for 95% confidence
```

See [references/statistics.md](references/statistics.md) for comprehensive validation methods.

### Phase 3: MQL5 Implementation

Structure for tick-based scalping EA:

```mql5
#property strict

input double RiskPercent = 1.0;        // Risk per trade (%)
input double MaxDailyDrawdown = 3.0;   // Max daily DD (%)
input int    MaxSpreadPoints = 20;     // Max spread filter

// Tick event handler - NOT OnBar
void OnTick() {
    if(!IsTradeAllowed()) return;
    if(!CheckRiskLimits()) return;
    if(!CheckSpread()) return;

    // Strategy logic here
    ProcessTickData();
}
```

See [references/mql5-patterns.md](references/mql5-patterns.md) for complete implementation patterns.

### Phase 4: Backtesting

1. **Python validation first**: Run on historical tick data
2. **MT5 Strategy Tester**: Use "Every tick based on real ticks" mode
3. **Walk-forward analysis**: Optimize on 70%, validate on 30%
4. **Monte Carlo simulation**: 1000+ iterations for robustness

See [references/backtesting.md](references/backtesting.md) for validation scripts.

## Risk Management Framework

### Position Sizing Methods

| Method | Use Case | Formula |
|--------|----------|---------|
| Fixed % | Conservative | `Lots = (Balance * RiskPct) / (SL_pips * PipValue)` |
| Kelly | Optimal growth | `f* = (bp - q) / b` where b=win/loss ratio |
| ATR-based | Volatility-adjusted | `Lots = RiskAmount / (ATR * Multiplier * PipValue)` |

### Risk Limits (Non-Negotiable)

```mql5
// Daily loss limit
input double MaxDailyLoss = 3.0;  // % of balance

// Correlation filter
input int MaxCorrelatedPositions = 2;

// Drawdown circuit breaker
input double MaxDrawdown = 10.0;  // Stop trading if exceeded
```

See [references/risk-management.md](references/risk-management.md) for implementation.

## Scalping vs Intraday Distinctions

### Scalping (Tick-Based)
- Entry/exit on tick events, NOT bar closes
- Target: 5-15 pips (Forex), $1-5 (Gold)
- Hold time: seconds to minutes
- Requires: Low latency, tight spreads, ECN broker
- Key metrics: Win rate > 55%, profit factor > 1.3

### Intraday
- Uses M5-H1 timeframes
- Target: 20-100 pips (Forex), $5-20 (Gold)
- Hold time: minutes to hours
- Close all positions before session end
- Key metrics: Sharpe > 1.0, max DD < 15%

## Asset-Specific Considerations

### Forex Majors (EURUSD, GBPUSD, USDJPY)
- Trade during London/NY overlap (13:00-17:00 UTC)
- Avoid news releases (NFP, FOMC, ECB)
- Typical spread: 0.5-1.5 pips

### Gold (XAUUSD)
- Higher volatility, wider stops required
- Correlates with USD index (inverse)
- Trade during London session for best liquidity
- Typical spread: 15-30 points

See [references/assets.md](references/assets.md) for detailed specifications.

## Performance Metrics Reference

| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | `(Returns - Rf) / StdDev(Returns)` | > 1.5 |
| Sortino Ratio | `(Returns - Rf) / DownsideDeviation` | > 2.0 |
| Calmar Ratio | `CAGR / MaxDrawdown` | > 1.0 |
| Profit Factor | `GrossProfit / GrossLoss` | > 1.5 |
| Z-Score | `(W - E[W]) / StdDev` | > 2.0 |

## Quick Reference: MQL5 Essentials

```mql5
// Get tick data
MqlTick tick;
SymbolInfoTick(_Symbol, tick);

// Calculate position size
double CalculateLots(double riskPercent, double slPoints) {
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * riskPercent / 100;
    return NormalizeDouble(riskAmount / (slPoints * tickValue), 2);
}

// Open position with risk management
bool OpenPosition(ENUM_ORDER_TYPE type, double lots, double sl, double tp) {
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lots;
    request.type = type;
    request.price = (type == ORDER_TYPE_BUY) ?
                    SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                    SymbolInfoDouble(_Symbol, SYMBOL_BID);
    request.sl = sl;
    request.tp = tp;
    request.deviation = 10;

    return OrderSend(request, result);
}
```

## File Resources

- **[references/statistics.md](references/statistics.md)**: Z-score, hypothesis testing, statistical validation
- **[references/mql5-patterns.md](references/mql5-patterns.md)**: EA templates, tick handling, order management
- **[references/risk-management.md](references/risk-management.md)**: Position sizing, drawdown control, correlation filters
- **[references/backtesting.md](references/backtesting.md)**: Python validation scripts, MT5 tester configuration
- **[references/assets.md](references/assets.md)**: Forex and Gold specifications, session times, spread analysis
- **[scripts/backtest_validator.py](scripts/backtest_validator.py)**: Python backtesting framework
- **[scripts/metrics_calculator.py](scripts/metrics_calculator.py)**: Performance metrics computation
- **[assets/ea_template.mq5](assets/ea_template.mq5)**: Base Expert Advisor template
