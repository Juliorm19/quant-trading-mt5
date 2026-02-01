# Backtesting and Validation

## Table of Contents
1. [Python Backtesting Framework](#python-backtesting-framework)
2. [MT5 Strategy Tester](#mt5-strategy-tester)
3. [Walk-Forward Analysis](#walk-forward-analysis)
4. [Out-of-Sample Testing](#out-of-sample-testing)
5. [Robustness Tests](#robustness-tests)

---

## Python Backtesting Framework

### Setup with MetaTrader5 Python Package

```python
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def initialize_mt5():
    """Initialize MT5 connection."""
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    return True

def get_tick_data(symbol, start_date, end_date):
    """Fetch tick data from MT5."""
    ticks = mt5.copy_ticks_range(
        symbol,
        start_date,
        end_date,
        mt5.COPY_TICKS_ALL
    )

    if ticks is None or len(ticks) == 0:
        print(f"No tick data for {symbol}")
        return None

    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

def get_ohlc_data(symbol, timeframe, start_date, end_date):
    """Fetch OHLC data from MT5."""
    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start_date,
        end_date
    )

    if rates is None or len(rates) == 0:
        print(f"No OHLC data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df
```

### Complete Backtest Engine

```python
class BacktestEngine:
    """
    Event-driven backtesting engine for tick or bar data.
    """

    def __init__(self, initial_balance=10000, commission=0.0001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.commission = commission  # Per lot per side

        self.positions = []
        self.closed_trades = []
        self.equity_curve = []

    def open_position(self, symbol, direction, lots, entry_price, sl, tp, timestamp):
        """Open a new position."""
        commission_cost = lots * self.commission * 2  # Round trip

        position = {
            'id': len(self.positions) + len(self.closed_trades),
            'symbol': symbol,
            'direction': direction,  # 1 for long, -1 for short
            'lots': lots,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'entry_time': timestamp,
            'commission': commission_cost,
            'pnl': 0
        }

        self.positions.append(position)
        self.balance -= commission_cost

        return position['id']

    def close_position(self, position_id, exit_price, timestamp, reason='manual'):
        """Close an existing position."""
        for i, pos in enumerate(self.positions):
            if pos['id'] == position_id:
                # Calculate P&L
                if pos['direction'] == 1:  # Long
                    pips = (exit_price - pos['entry_price']) / 0.0001
                else:  # Short
                    pips = (pos['entry_price'] - exit_price) / 0.0001

                # Standard lot = $10 per pip for most pairs
                pnl = pips * 10 * pos['lots']

                pos['exit_price'] = exit_price
                pos['exit_time'] = timestamp
                pos['pnl'] = pnl
                pos['exit_reason'] = reason

                self.balance += pnl
                self.closed_trades.append(pos)
                self.positions.pop(i)

                return pnl

        return 0

    def update_equity(self, current_prices, timestamp):
        """Update equity with unrealized P&L."""
        unrealized = 0

        for pos in self.positions:
            symbol = pos['symbol']
            if symbol in current_prices:
                price = current_prices[symbol]

                if pos['direction'] == 1:
                    pips = (price - pos['entry_price']) / 0.0001
                else:
                    pips = (pos['entry_price'] - price) / 0.0001

                unrealized += pips * 10 * pos['lots']

        self.equity = self.balance + unrealized
        self.equity_curve.append({
            'time': timestamp,
            'equity': self.equity,
            'balance': self.balance
        })

    def check_stops(self, current_prices, timestamp):
        """Check and execute stop loss and take profit."""
        positions_to_close = []

        for pos in self.positions:
            symbol = pos['symbol']
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]

            if pos['direction'] == 1:  # Long
                if pos['sl'] and price <= pos['sl']:
                    positions_to_close.append((pos['id'], pos['sl'], 'sl'))
                elif pos['tp'] and price >= pos['tp']:
                    positions_to_close.append((pos['id'], pos['tp'], 'tp'))
            else:  # Short
                if pos['sl'] and price >= pos['sl']:
                    positions_to_close.append((pos['id'], pos['sl'], 'sl'))
                elif pos['tp'] and price <= pos['tp']:
                    positions_to_close.append((pos['id'], pos['tp'], 'tp'))

        for pos_id, exit_price, reason in positions_to_close:
            self.close_position(pos_id, exit_price, timestamp, reason)

    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.closed_trades:
            return None

        returns = [t['pnl'] for t in self.closed_trades]
        returns = np.array(returns)

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        total_return = (self.balance - self.initial_balance) / self.initial_balance

        # Sharpe Ratio (assuming daily returns, 252 trading days)
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino Ratio
        downside = returns[returns < 0]
        if len(downside) > 1:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Maximum Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        if equity_values:
            peak = np.maximum.accumulate(equity_values)
            drawdown = (np.array(equity_values) - peak) / peak
            max_dd = np.min(drawdown)
        else:
            max_dd = 0

        # Calmar Ratio
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0

        # Profit Factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_return': total_return * 100,
            'total_trades': len(self.closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(returns) * 100 if len(returns) > 0 else 0,
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'largest_win': np.max(wins) if len(wins) > 0 else 0,
            'largest_loss': np.min(losses) if len(losses) > 0 else 0,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'calmar_ratio': calmar,
            'profit_factor': profit_factor,
            'final_balance': self.balance
        }
```

### Example Strategy Backtest

```python
def backtest_zscore_mean_reversion(symbol, data, lookback=20, entry_z=2.0, exit_z=0.5):
    """
    Backtest a z-score mean reversion strategy.

    Args:
        symbol: Trading symbol
        data: DataFrame with 'close' column
        lookback: Period for calculating mean/std
        entry_z: Z-score threshold for entry
        exit_z: Z-score threshold for exit
    """
    engine = BacktestEngine(initial_balance=10000)

    # Calculate rolling z-score
    data['mean'] = data['close'].rolling(lookback).mean()
    data['std'] = data['close'].rolling(lookback).std()
    data['zscore'] = (data['close'] - data['mean']) / data['std']

    position = None
    risk_per_trade = 0.01  # 1%

    for i in range(lookback, len(data)):
        timestamp = data.index[i]
        price = data['close'].iloc[i]
        zscore = data['zscore'].iloc[i]

        # Update engine
        engine.update_equity({symbol: price}, timestamp)
        engine.check_stops({symbol: price}, timestamp)

        # Check for existing position
        has_position = len(engine.positions) > 0

        if not has_position:
            # Entry signals
            if zscore < -entry_z:  # Oversold - Buy
                sl = price - data['std'].iloc[i] * 3
                tp = data['mean'].iloc[i]
                lots = calculate_lots(engine.balance, risk_per_trade, price - sl)

                engine.open_position(symbol, 1, lots, price, sl, tp, timestamp)

            elif zscore > entry_z:  # Overbought - Sell
                sl = price + data['std'].iloc[i] * 3
                tp = data['mean'].iloc[i]
                lots = calculate_lots(engine.balance, risk_per_trade, sl - price)

                engine.open_position(symbol, -1, lots, price, sl, tp, timestamp)

        else:
            # Exit signals
            pos = engine.positions[0]
            if pos['direction'] == 1 and zscore > -exit_z:
                engine.close_position(pos['id'], price, timestamp, 'signal')
            elif pos['direction'] == -1 and zscore < exit_z:
                engine.close_position(pos['id'], price, timestamp, 'signal')

    # Close any remaining positions
    if engine.positions:
        final_price = data['close'].iloc[-1]
        for pos in engine.positions[:]:
            engine.close_position(pos['id'], final_price, data.index[-1], 'end')

    return engine.get_performance_metrics(), engine.equity_curve

def calculate_lots(balance, risk_pct, sl_distance, pip_value=10):
    """Calculate position size based on risk."""
    risk_amount = balance * risk_pct
    sl_pips = sl_distance / 0.0001
    lots = risk_amount / (sl_pips * pip_value)
    return round(lots, 2)
```

---

## MT5 Strategy Tester

### Optimal Settings for Scalping

```
Mode: Every tick based on real ticks
Period: M1 or custom ticks
Deposit: Match intended live account
Leverage: Match broker leverage
Spread: Variable (from history)
```

### Tester Configuration in EA

```mql5
// Optimization parameters
input group "=== Optimization Ranges ==="
input int FastPeriod_Start = 3;       // Fast period start
input int FastPeriod_Step = 1;        // Fast period step
input int FastPeriod_End = 20;        // Fast period end

input int SlowPeriod_Start = 10;      // Slow period start
input int SlowPeriod_Step = 5;        // Slow period step
input int SlowPeriod_End = 100;       // Slow period end

// OnTester - Custom optimization criterion
double OnTester() {
    // Get statistics
    double profit = TesterStatistics(STAT_PROFIT);
    double trades = TesterStatistics(STAT_TRADES);
    double drawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
    double sharpe = TesterStatistics(STAT_SHARPE_RATIO);
    double profitFactor = TesterStatistics(STAT_PROFIT_FACTOR);

    // Minimum trade requirement
    if(trades < 100) return 0;

    // Custom fitness function
    // Prioritize Sharpe ratio and profit factor while penalizing drawdown
    double fitness = sharpe * profitFactor / (1 + drawdown / 10);

    return fitness;
}

// Custom frame processing for genetic optimization
void OnTesterPass() {
    ulong pass;
    string name;
    long id;
    double value;
    double arr[];

    FrameFirst();
    FrameFilter("", 0);

    while(FrameNext(pass, name, id, value, arr)) {
        Print("Pass: ", pass, " Fitness: ", value);
    }
}
```

### Report Export

```mql5
void ExportDetailedReport() {
    string filename = "backtest_report_" +
                      TimeToString(TimeCurrent(), TIME_DATE) + ".csv";

    int handle = FileOpen(filename, FILE_WRITE | FILE_CSV);
    if(handle == INVALID_HANDLE) return;

    // Header
    FileWrite(handle, "Ticket", "Symbol", "Type", "Volume", "OpenTime",
              "OpenPrice", "SL", "TP", "CloseTime", "ClosePrice",
              "Profit", "Commission", "Swap");

    // Write deals
    HistorySelect(0, TimeCurrent());

    for(int i = 0; i < HistoryDealsTotal(); i++) {
        ulong ticket = HistoryDealGetTicket(i);

        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
            FileWrite(handle,
                ticket,
                HistoryDealGetString(ticket, DEAL_SYMBOL),
                EnumToString((ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket, DEAL_TYPE)),
                HistoryDealGetDouble(ticket, DEAL_VOLUME),
                TimeToString((datetime)HistoryDealGetInteger(ticket, DEAL_TIME)),
                HistoryDealGetDouble(ticket, DEAL_PRICE),
                0, 0,  // SL/TP not directly available
                "",
                HistoryDealGetDouble(ticket, DEAL_PRICE),
                HistoryDealGetDouble(ticket, DEAL_PROFIT),
                HistoryDealGetDouble(ticket, DEAL_COMMISSION),
                HistoryDealGetDouble(ticket, DEAL_SWAP)
            );
        }
    }

    FileClose(handle);
}
```

---

## Walk-Forward Analysis

### Python Implementation

```python
def walk_forward_analysis(data, strategy_func, optimize_func,
                          train_size=0.7, n_splits=5, params_grid=None):
    """
    Perform walk-forward optimization.

    Args:
        data: Full dataset
        strategy_func: Function that runs strategy given params
        optimize_func: Function that optimizes parameters on training data
        train_size: Proportion of each split for training
        n_splits: Number of walk-forward periods
        params_grid: Parameter grid for optimization
    """
    results = []
    split_size = len(data) // n_splits

    for i in range(n_splits):
        # Define train/test periods
        start_idx = i * split_size
        end_idx = start_idx + split_size

        train_end = start_idx + int(split_size * train_size)

        train_data = data.iloc[start_idx:train_end]
        test_data = data.iloc[train_end:end_idx]

        print(f"Split {i+1}/{n_splits}")
        print(f"  Train: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"  Test:  {test_data.index[0]} to {test_data.index[-1]}")

        # Optimize on training data
        best_params = optimize_func(train_data, params_grid)
        print(f"  Best params: {best_params}")

        # Test on out-of-sample
        metrics, equity = strategy_func(test_data, **best_params)
        print(f"  OOS Results: Sharpe={metrics['sharpe_ratio']:.2f}, "
              f"Return={metrics['total_return']:.2f}%")

        results.append({
            'split': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'best_params': best_params,
            'metrics': metrics
        })

    # Aggregate results
    avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results])
    avg_return = np.mean([r['metrics']['total_return'] for r in results])
    avg_dd = np.mean([r['metrics']['max_drawdown'] for r in results])

    print(f"\nWalk-Forward Summary:")
    print(f"  Avg Sharpe: {avg_sharpe:.2f}")
    print(f"  Avg Return: {avg_return:.2f}%")
    print(f"  Avg MaxDD:  {avg_dd:.2f}%")

    return results
```

### Optimization Function Example

```python
from itertools import product

def optimize_zscore_params(data, params_grid):
    """
    Grid search optimization for z-score strategy.
    """
    if params_grid is None:
        params_grid = {
            'lookback': [10, 15, 20, 30],
            'entry_z': [1.5, 2.0, 2.5],
            'exit_z': [0.3, 0.5, 0.7]
        }

    best_sharpe = -np.inf
    best_params = None

    # Generate all combinations
    keys = params_grid.keys()
    values = params_grid.values()

    for combination in product(*values):
        params = dict(zip(keys, combination))

        # Run backtest
        metrics, _ = backtest_zscore_mean_reversion(
            'EURUSD', data,
            lookback=params['lookback'],
            entry_z=params['entry_z'],
            exit_z=params['exit_z']
        )

        if metrics and metrics['sharpe_ratio'] > best_sharpe:
            if metrics['total_trades'] >= 30:  # Minimum trades
                best_sharpe = metrics['sharpe_ratio']
                best_params = params.copy()

    return best_params
```

---

## Out-of-Sample Testing

### Data Splitting Protocol

```python
def split_data_for_validation(data, train_pct=0.6, validation_pct=0.2, test_pct=0.2):
    """
    Split data into train/validation/test sets.

    - Train: Develop strategy
    - Validation: Tune parameters
    - Test: Final unbiased evaluation
    """
    n = len(data)

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + validation_pct))

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    print(f"Train:      {len(train)} bars ({train.index[0]} to {train.index[-1]})")
    print(f"Validation: {len(validation)} bars ({validation.index[0]} to {validation.index[-1]})")
    print(f"Test:       {len(test)} bars ({test.index[0]} to {test.index[-1]})")

    return train, validation, test
```

### Degradation Analysis

```python
def analyze_performance_degradation(in_sample_metrics, out_sample_metrics):
    """
    Analyze how much performance degrades out-of-sample.
    High degradation indicates overfitting.
    """
    degradation = {}

    for key in ['sharpe_ratio', 'profit_factor', 'win_rate', 'total_return']:
        if key in in_sample_metrics and key in out_sample_metrics:
            is_val = in_sample_metrics[key]
            oos_val = out_sample_metrics[key]

            if is_val != 0:
                pct_change = (oos_val - is_val) / abs(is_val) * 100
            else:
                pct_change = 0

            degradation[key] = {
                'in_sample': is_val,
                'out_sample': oos_val,
                'degradation_pct': pct_change
            }

    # Flag if degradation exceeds thresholds
    warnings = []

    if degradation.get('sharpe_ratio', {}).get('degradation_pct', 0) < -30:
        warnings.append("Sharpe ratio degraded > 30% - possible overfitting")

    if degradation.get('profit_factor', {}).get('degradation_pct', 0) < -40:
        warnings.append("Profit factor degraded > 40% - check parameter sensitivity")

    if degradation.get('win_rate', {}).get('degradation_pct', 0) < -20:
        warnings.append("Win rate degraded > 20% - entry logic may be curve-fitted")

    return degradation, warnings
```

---

## Robustness Tests

### Parameter Sensitivity

```python
def parameter_sensitivity_analysis(data, strategy_func, base_params, param_ranges):
    """
    Test how sensitive results are to parameter changes.
    Robust strategies show gradual performance changes.
    """
    results = {}

    for param_name, variations in param_ranges.items():
        results[param_name] = []

        for value in variations:
            test_params = base_params.copy()
            test_params[param_name] = value

            metrics, _ = strategy_func(data, **test_params)

            results[param_name].append({
                'value': value,
                'sharpe': metrics['sharpe_ratio'] if metrics else 0,
                'return': metrics['total_return'] if metrics else 0,
                'trades': metrics['total_trades'] if metrics else 0
            })

    # Analyze stability
    stability_scores = {}
    for param_name, values in results.items():
        sharpes = [v['sharpe'] for v in values]
        if len(sharpes) > 1:
            stability_scores[param_name] = {
                'mean': np.mean(sharpes),
                'std': np.std(sharpes),
                'cv': np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')
            }

    return results, stability_scores
```

### Monte Carlo Robustness

```python
def monte_carlo_robustness(trades, n_simulations=10000):
    """
    Monte Carlo simulation to test strategy robustness.
    Tests: trade reordering, random removal, slippage simulation.
    """
    results = {
        'reorder': [],
        'removal': [],
        'slippage': []
    }

    pnls = np.array([t['pnl'] for t in trades])

    for _ in range(n_simulations):
        # Test 1: Random reordering
        shuffled = np.random.permutation(pnls)
        equity = np.cumprod(1 + shuffled / 10000)
        max_dd = calculate_max_drawdown(equity)
        final_return = equity[-1] - 1
        results['reorder'].append({'return': final_return, 'max_dd': max_dd})

        # Test 2: Random 10% removal
        mask = np.random.random(len(pnls)) > 0.1
        reduced = pnls[mask]
        if len(reduced) > 0:
            equity = np.cumprod(1 + reduced / 10000)
            max_dd = calculate_max_drawdown(equity)
            final_return = equity[-1] - 1
            results['removal'].append({'return': final_return, 'max_dd': max_dd})

        # Test 3: Random slippage (0-2 pips)
        slippage = np.random.uniform(-20, 0, len(pnls))  # $0-20 adverse slippage
        adjusted = pnls + slippage
        equity = np.cumprod(1 + adjusted / 10000)
        max_dd = calculate_max_drawdown(equity)
        final_return = equity[-1] - 1
        results['slippage'].append({'return': final_return, 'max_dd': max_dd})

    # Summarize
    summary = {}
    for test_name, test_results in results.items():
        returns = [r['return'] for r in test_results]
        drawdowns = [r['max_dd'] for r in test_results]

        summary[test_name] = {
            'return_mean': np.mean(returns) * 100,
            'return_5th': np.percentile(returns, 5) * 100,
            'return_95th': np.percentile(returns, 95) * 100,
            'max_dd_mean': np.mean(drawdowns) * 100,
            'max_dd_95th': np.percentile(drawdowns, 95) * 100,
            'prob_profitable': np.mean(np.array(returns) > 0) * 100
        }

    return summary

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)
```

### Data Snooping Bias Test

```python
def data_snooping_test(data, strategy_func, n_random_strategies=1000):
    """
    Test if strategy is better than random.
    Generates random strategies and compares performance.
    """
    # Run actual strategy
    actual_metrics, _ = strategy_func(data)
    actual_sharpe = actual_metrics['sharpe_ratio']

    # Generate random strategies with similar characteristics
    random_sharpes = []

    for _ in range(n_random_strategies):
        # Random entry/exit signals
        n_trades = actual_metrics['total_trades']
        random_returns = np.random.normal(0, 0.01, n_trades)

        sharpe = np.mean(random_returns) / np.std(random_returns) * np.sqrt(252)
        random_sharpes.append(sharpe)

    # Calculate p-value
    better_than_random = np.sum(np.array(random_sharpes) >= actual_sharpe)
    p_value = better_than_random / n_random_strategies

    return {
        'actual_sharpe': actual_sharpe,
        'random_mean': np.mean(random_sharpes),
        'random_95th': np.percentile(random_sharpes, 95),
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```
