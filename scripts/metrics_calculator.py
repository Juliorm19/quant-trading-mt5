#!/usr/bin/env python3
"""
Performance Metrics Calculator for Trading Strategies
Comprehensive statistical analysis for quant trading validation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import json


@dataclass
class TradeMetrics:
    """Container for trade-level metrics."""
    total_return: float
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade: float
    win_loss_ratio: float
    expectancy: float


@dataclass
class RiskMetrics:
    """Container for risk-adjusted metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    ulcer_index: float
    var_95: float
    cvar_95: float


@dataclass
class StatisticalMetrics:
    """Container for statistical validation metrics."""
    z_score: float
    p_value: float
    t_statistic: float
    confidence_interval_95: Tuple[float, float]
    is_significant: bool
    required_trades_95: int


class MetricsCalculator:
    """
    Comprehensive performance metrics calculator.
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        self.rf_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_all(self, returns: np.ndarray,
                      equity_curve: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate all metrics from returns.

        Args:
            returns: Array of period returns (not cumulative)
            equity_curve: Optional equity curve array

        Returns:
            Dictionary with all metrics
        """
        if equity_curve is None:
            equity_curve = self._build_equity(returns)

        trade_metrics = self._trade_metrics(returns)
        risk_metrics = self._risk_metrics(returns, equity_curve)
        stat_metrics = self._statistical_metrics(returns)

        return {
            'trade': trade_metrics.__dict__,
            'risk': risk_metrics.__dict__,
            'statistical': {
                **stat_metrics.__dict__,
                'confidence_interval_95': list(stat_metrics.confidence_interval_95)
            }
        }

    def _build_equity(self, returns: np.ndarray, initial: float = 10000) -> np.ndarray:
        """Build equity curve from returns."""
        return initial * np.cumprod(1 + returns)

    def _trade_metrics(self, returns: np.ndarray) -> TradeMetrics:
        """Calculate trade-level metrics."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        total_return = np.sum(returns)
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0

        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Expectancy: (Win% × Avg Win) - (Loss% × |Avg Loss|)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return TradeMetrics(
            total_return=total_return,
            num_trades=len(returns),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=np.max(wins) if len(wins) > 0 else 0,
            largest_loss=np.min(losses) if len(losses) > 0 else 0,
            avg_trade=np.mean(returns),
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy
        )

    def _risk_metrics(self, returns: np.ndarray,
                      equity_curve: np.ndarray) -> RiskMetrics:
        """Calculate risk-adjusted metrics."""
        rf_period = self.rf_rate / self.periods_per_year
        excess_returns = returns - rf_period

        # Sharpe Ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.sqrt(self.periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
        else:
            sharpe = 0

        # Sortino Ratio (downside deviation)
        downside = returns[returns < rf_period]
        if len(downside) > 1:
            downside_std = np.std(downside)
            sortino = np.sqrt(self.periods_per_year) * np.mean(excess_returns) / downside_std
        else:
            sortino = sharpe

        # Drawdown analysis
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = np.min(drawdown)
        avg_dd = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0

        # Max drawdown duration
        in_drawdown = drawdown < 0
        dd_duration = 0
        max_dd_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Calmar Ratio
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        cagr = (1 + total_return) ** (self.periods_per_year / len(returns)) - 1
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Ulcer Index (RMS of drawdowns)
        ulcer = np.sqrt(np.mean(drawdown ** 2))

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])

        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            ulcer_index=ulcer,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def _statistical_metrics(self, returns: np.ndarray) -> StatisticalMetrics:
        """Calculate statistical significance metrics."""
        n = len(returns)

        if n < 2:
            return StatisticalMetrics(
                z_score=0, p_value=1, t_statistic=0,
                confidence_interval_95=(0, 0),
                is_significant=False,
                required_trades_95=100
            )

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        se = std / np.sqrt(n)

        # Z-score (testing if mean differs from 0)
        z_score = mean / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # T-statistic
        t_stat, t_pvalue = stats.ttest_1samp(returns, 0)

        # 95% confidence interval for mean return
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        # Required trades for statistical significance
        # n = (z^2 * sigma^2) / E^2 where E is margin of error
        if mean != 0:
            margin = abs(mean) * 0.1  # 10% margin of error
            required = int(np.ceil((1.96 ** 2 * std ** 2) / (margin ** 2)))
        else:
            required = 100

        return StatisticalMetrics(
            z_score=z_score,
            p_value=p_value,
            t_statistic=t_stat,
            confidence_interval_95=(ci_low, ci_high),
            is_significant=p_value < 0.05,
            required_trades_95=required
        )


class StrategyComparator:
    """
    Compare multiple strategies statistically.
    """

    @staticmethod
    def compare_sharpe(returns_a: np.ndarray, returns_b: np.ndarray) -> Dict:
        """
        Compare Sharpe ratios of two strategies using Ledoit-Wolf test.

        Args:
            returns_a: Returns of strategy A
            returns_b: Returns of strategy B

        Returns:
            Comparison results
        """
        n = len(returns_a)

        sharpe_a = np.mean(returns_a) / np.std(returns_a) * np.sqrt(252)
        sharpe_b = np.mean(returns_b) / np.std(returns_b) * np.sqrt(252)

        # Paired difference
        diff = returns_a - returns_b
        se_diff = np.std(diff) / np.sqrt(n)

        z = (np.mean(returns_a) - np.mean(returns_b)) / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            'sharpe_a': sharpe_a,
            'sharpe_b': sharpe_b,
            'difference': sharpe_a - sharpe_b,
            'z_statistic': z,
            'p_value': p_value,
            'a_significantly_better': sharpe_a > sharpe_b and p_value < 0.05,
            'b_significantly_better': sharpe_b > sharpe_a and p_value < 0.05
        }

    @staticmethod
    def bootstrap_comparison(returns_a: np.ndarray, returns_b: np.ndarray,
                             n_bootstrap: int = 10000) -> Dict:
        """
        Bootstrap comparison of two strategies.
        """
        n = len(returns_a)
        sharpe_diffs = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            sample_a = returns_a[idx]
            sample_b = returns_b[idx]

            sharpe_a = np.mean(sample_a) / np.std(sample_a) * np.sqrt(252)
            sharpe_b = np.mean(sample_b) / np.std(sample_b) * np.sqrt(252)

            sharpe_diffs.append(sharpe_a - sharpe_b)

        sharpe_diffs = np.array(sharpe_diffs)

        return {
            'mean_difference': np.mean(sharpe_diffs),
            'ci_95_lower': np.percentile(sharpe_diffs, 2.5),
            'ci_95_upper': np.percentile(sharpe_diffs, 97.5),
            'prob_a_better': np.mean(sharpe_diffs > 0),
            'significant': not (np.percentile(sharpe_diffs, 2.5) <= 0 <= np.percentile(sharpe_diffs, 97.5))
        }


def calculate_from_trades(trades: List[Dict],
                          initial_balance: float = 10000) -> Dict:
    """
    Calculate all metrics from a list of trades.

    Args:
        trades: List of dicts with 'pnl' key
        initial_balance: Starting balance

    Returns:
        Complete metrics dictionary
    """
    pnls = np.array([t['pnl'] for t in trades])
    returns = pnls / initial_balance

    # Build equity curve
    equity = [initial_balance]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    equity = np.array(equity[1:])

    calc = MetricsCalculator()
    return calc.calculate_all(returns, equity)


def print_metrics_report(metrics: Dict):
    """Print formatted metrics report."""
    print("\n" + "=" * 60)
    print("STRATEGY PERFORMANCE REPORT")
    print("=" * 60)

    print("\n--- Trade Metrics ---")
    tm = metrics['trade']
    print(f"  Total Return:     ${tm['total_return']:.2f}")
    print(f"  Num Trades:       {tm['num_trades']}")
    print(f"  Win Rate:         {tm['win_rate']*100:.1f}%")
    print(f"  Profit Factor:    {tm['profit_factor']:.2f}")
    print(f"  Avg Win:          ${tm['avg_win']:.2f}")
    print(f"  Avg Loss:         ${tm['avg_loss']:.2f}")
    print(f"  Win/Loss Ratio:   {tm['win_loss_ratio']:.2f}")
    print(f"  Expectancy:       ${tm['expectancy']:.2f}")
    print(f"  Largest Win:      ${tm['largest_win']:.2f}")
    print(f"  Largest Loss:     ${tm['largest_loss']:.2f}")

    print("\n--- Risk Metrics ---")
    rm = metrics['risk']
    print(f"  Sharpe Ratio:     {rm['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:    {rm['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:     {rm['calmar_ratio']:.2f}")
    print(f"  Max Drawdown:     {rm['max_drawdown']*100:.1f}%")
    print(f"  Avg Drawdown:     {rm['avg_drawdown']*100:.1f}%")
    print(f"  Max DD Duration:  {rm['max_drawdown_duration']} periods")
    print(f"  Ulcer Index:      {rm['ulcer_index']:.4f}")
    print(f"  VaR (95%):        {rm['var_95']*100:.2f}%")
    print(f"  CVaR (95%):       {rm['cvar_95']*100:.2f}%")

    print("\n--- Statistical Validation ---")
    sm = metrics['statistical']
    print(f"  Z-Score:          {sm['z_score']:.2f}")
    print(f"  P-Value:          {sm['p_value']:.4f}")
    print(f"  T-Statistic:      {sm['t_statistic']:.2f}")
    print(f"  95% CI:           [{sm['confidence_interval_95'][0]*100:.2f}%, {sm['confidence_interval_95'][1]*100:.2f}%]")
    print(f"  Significant:      {'Yes' if sm['is_significant'] else 'No'} (p < 0.05)")
    print(f"  Min Trades Req:   {sm['required_trades_95']}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Example usage with synthetic data
    print("Metrics Calculator Demo")
    print("-" * 40)

    # Generate synthetic strategy returns
    np.random.seed(42)
    n_trades = 200

    # Strategy with slight edge
    win_rate = 0.55
    avg_win = 0.015  # 1.5%
    avg_loss = -0.012  # -1.2%

    returns = []
    for _ in range(n_trades):
        if np.random.random() < win_rate:
            returns.append(avg_win * (0.5 + np.random.random()))
        else:
            returns.append(avg_loss * (0.5 + np.random.random()))

    returns = np.array(returns)

    # Calculate metrics
    calc = MetricsCalculator()
    equity = 10000 * np.cumprod(1 + returns)
    metrics = calc.calculate_all(returns, equity)

    print_metrics_report(metrics)

    # Also output as JSON for programmatic use
    print("\nJSON Output:")
    print(json.dumps(metrics, indent=2))
