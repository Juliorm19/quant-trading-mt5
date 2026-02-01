# Statistical Validation for Trading Strategies

## Table of Contents
1. [Z-Score Analysis](#z-score-analysis)
2. [Hypothesis Testing](#hypothesis-testing)
3. [Performance Metrics](#performance-metrics)
4. [Statistical Significance](#statistical-significance)
5. [Monte Carlo Simulation](#monte-carlo-simulation)

---

## Z-Score Analysis

### Purpose
Determine if strategy returns are statistically different from random chance.

### Calculation

```python
import numpy as np
from scipy import stats

def calculate_zscore(returns, benchmark_mean=0):
    """
    Calculate z-score for strategy returns.

    Args:
        returns: Array of trade returns
        benchmark_mean: Expected return under null hypothesis (usually 0)

    Returns:
        z_score, p_value
    """
    n = len(returns)
    sample_mean = np.mean(returns)
    sample_std = np.std(returns, ddof=1)

    z_score = (sample_mean - benchmark_mean) / (sample_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return z_score, p_value
```

### Interpretation
| Z-Score | Confidence | Action |
|---------|------------|--------|
| < 1.65 | < 90% | Reject strategy |
| 1.65-1.96 | 90-95% | Weak evidence, more data needed |
| 1.96-2.58 | 95-99% | Acceptable for live testing |
| > 2.58 | > 99% | Strong evidence of edge |

### Win/Loss Streak Z-Score
Detect non-random clustering of wins/losses:

```python
def streak_zscore(trades):
    """
    Test if win/loss streaks are random.
    Non-random streaks may indicate regime dependency.
    """
    wins = np.array([1 if t > 0 else 0 for t in trades])
    n = len(wins)
    n1 = np.sum(wins)  # Number of wins
    n2 = n - n1        # Number of losses

    # Count runs
    runs = 1
    for i in range(1, n):
        if wins[i] != wins[i-1]:
            runs += 1

    # Expected runs and std under null
    expected_runs = (2 * n1 * n2) / n + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1)))

    z = (runs - expected_runs) / std_runs
    return z
```

---

## Hypothesis Testing

### Trade-Level Testing

```python
def ttest_strategy_returns(returns, null_mean=0):
    """
    One-sample t-test for strategy returns.
    H0: mean return = null_mean
    H1: mean return != null_mean
    """
    t_stat, p_value = stats.ttest_1samp(returns, null_mean)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_95': p_value < 0.05,
        'significant_99': p_value < 0.01
    }
```

### Comparing Two Strategies

```python
def compare_strategies(returns_a, returns_b):
    """
    Welch's t-test for comparing two strategies.
    Does not assume equal variances.
    """
    t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'strategy_a_better': np.mean(returns_a) > np.mean(returns_b) and p_value < 0.05
    }
```

---

## Performance Metrics

### Complete Metrics Suite

```python
def calculate_all_metrics(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Array of period returns (not cumulative)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily, 252*24 for hourly)
    """
    returns = np.array(returns)
    rf_period = risk_free_rate / periods_per_year

    # Basic stats
    total_return = np.prod(1 + returns) - 1
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    # Sharpe Ratio (annualized)
    excess_returns = returns - rf_period
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < rf_period]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # Calmar Ratio
    cagr = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.inf

    # Win Rate and Profit Factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns)
    gross_profit = np.sum(wins)
    gross_loss = abs(np.sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Average Win/Loss
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward': risk_reward,
        'num_trades': len(returns)
    }
```

### Metric Targets by Strategy Type

| Metric | Scalping | Intraday | Swing |
|--------|----------|----------|-------|
| Sharpe Ratio | > 1.5 | > 1.0 | > 0.8 |
| Sortino Ratio | > 2.0 | > 1.5 | > 1.0 |
| Max Drawdown | < 10% | < 15% | < 20% |
| Win Rate | > 55% | > 50% | > 45% |
| Profit Factor | > 1.3 | > 1.5 | > 1.8 |
| Calmar Ratio | > 1.5 | > 1.0 | > 0.8 |

---

## Statistical Significance

### Minimum Sample Size

```python
def min_trades_required(expected_win_rate, margin_of_error=0.05, confidence=0.95):
    """
    Calculate minimum trades needed for statistically significant results.

    Uses: n = (z^2 * p * (1-p)) / e^2
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    p = expected_win_rate
    e = margin_of_error

    n = (z**2 * p * (1 - p)) / (e**2)
    return int(np.ceil(n))

# Example: 55% win rate, 5% margin, 95% confidence
# Requires ~380 trades minimum
```

### Bootstrap Confidence Intervals

```python
def bootstrap_sharpe(returns, n_bootstrap=10000, confidence=0.95):
    """
    Bootstrap confidence interval for Sharpe ratio.
    """
    sharpes = []
    n = len(returns)

    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        sharpe = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
        sharpes.append(sharpe)

    lower = np.percentile(sharpes, (1 - confidence) / 2 * 100)
    upper = np.percentile(sharpes, (1 + confidence) / 2 * 100)

    return {
        'mean_sharpe': np.mean(sharpes),
        'lower_bound': lower,
        'upper_bound': upper,
        'significant': lower > 0  # Sharpe CI doesn't include 0
    }
```

---

## Monte Carlo Simulation

### Trade Sequence Randomization

```python
def monte_carlo_drawdown(returns, n_simulations=10000):
    """
    Estimate distribution of maximum drawdown through random reordering.
    Tests if observed drawdown is due to skill or luck.
    """
    observed_dd = calculate_max_drawdown(returns)
    simulated_dds = []

    for _ in range(n_simulations):
        shuffled = np.random.permutation(returns)
        dd = calculate_max_drawdown(shuffled)
        simulated_dds.append(dd)

    # Percentile of observed drawdown
    percentile = np.mean(np.array(simulated_dds) >= observed_dd) * 100

    return {
        'observed_drawdown': observed_dd,
        'mean_simulated': np.mean(simulated_dds),
        'worst_case_95': np.percentile(simulated_dds, 5),  # 5th percentile is worst
        'percentile': percentile
    }

def calculate_max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return np.min(drawdowns)
```

### Equity Curve Simulation

```python
def simulate_equity_curves(win_rate, avg_win, avg_loss, n_trades=1000, n_curves=1000):
    """
    Simulate possible equity curves given strategy parameters.
    Useful for setting realistic expectations.
    """
    curves = []

    for _ in range(n_curves):
        trades = []
        for _ in range(n_trades):
            if np.random.random() < win_rate:
                trades.append(avg_win)
            else:
                trades.append(avg_loss)

        equity = np.cumprod(1 + np.array(trades))
        curves.append(equity)

    curves = np.array(curves)

    return {
        'median_final': np.median(curves[:, -1]),
        'percentile_5': np.percentile(curves[:, -1], 5),
        'percentile_95': np.percentile(curves[:, -1], 95),
        'probability_profit': np.mean(curves[:, -1] > 1.0),
        'curves': curves  # For visualization
    }
```
