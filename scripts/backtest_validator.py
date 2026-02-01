#!/usr/bin/env python3
"""
Backtest Validator for MT5 Trading Strategies
Validates strategy performance with comprehensive statistical analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class BacktestValidator:
    """
    Validates backtest results against quality criteria.
    """

    def __init__(self, min_trades: int = 100, min_sharpe: float = 1.0,
                 max_drawdown: float = 0.15, min_profit_factor: float = 1.3):
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_profit_factor = min_profit_factor

    def validate(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """
        Validate backtest results against criteria.

        Args:
            trades: List of trade dictionaries with 'pnl', 'entry_time', 'exit_time'
            equity_curve: List of equity values over time

        Returns:
            Validation report dictionary
        """
        if len(trades) < self.min_trades:
            return {
                'valid': False,
                'reason': f'Insufficient trades: {len(trades)} < {self.min_trades}',
                'metrics': None
            }

        metrics = self._calculate_metrics(trades, equity_curve)

        # Validation checks
        issues = []

        if metrics['sharpe_ratio'] < self.min_sharpe:
            issues.append(f"Sharpe ratio {metrics['sharpe_ratio']:.2f} < {self.min_sharpe}")

        if abs(metrics['max_drawdown']) > self.max_drawdown:
            issues.append(f"Max drawdown {metrics['max_drawdown']*100:.1f}% > {self.max_drawdown*100:.1f}%")

        if metrics['profit_factor'] < self.min_profit_factor:
            issues.append(f"Profit factor {metrics['profit_factor']:.2f} < {self.min_profit_factor}")

        # Statistical significance check
        z_score = self._calculate_zscore(trades)
        if z_score < 2.0:
            issues.append(f"Z-score {z_score:.2f} < 2.0 (not statistically significant)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metrics': metrics,
            'z_score': z_score
        }

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate comprehensive performance metrics."""
        pnls = np.array([t['pnl'] for t in trades])

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        # Basic metrics
        total_return = np.sum(pnls)
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized, assuming daily returns)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio
        downside = pnls[pnls < 0]
        if len(downside) > 1:
            sortino = np.mean(pnls) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Maximum drawdown
        if len(equity_curve) > 0:
            equity = np.array(equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_dd = np.min(drawdown)
        else:
            max_dd = 0

        # Calmar ratio
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'largest_win': np.max(wins) if len(wins) > 0 else 0,
            'largest_loss': np.min(losses) if len(losses) > 0 else 0,
            'avg_trade': np.mean(pnls)
        }

    def _calculate_zscore(self, trades: List[Dict]) -> float:
        """Calculate z-score to test if returns differ from zero."""
        pnls = np.array([t['pnl'] for t in trades])
        n = len(pnls)

        if n < 2:
            return 0

        mean = np.mean(pnls)
        std = np.std(pnls, ddof=1)

        if std == 0:
            return 0

        z = mean / (std / np.sqrt(n))
        return z


class WalkForwardValidator:
    """
    Performs walk-forward analysis to detect overfitting.
    """

    def __init__(self, n_splits: int = 5, train_ratio: float = 0.7):
        self.n_splits = n_splits
        self.train_ratio = train_ratio

    def analyze(self, data: pd.DataFrame,
                strategy_func,
                optimize_func,
                param_grid: Dict) -> Dict:
        """
        Perform walk-forward optimization and validation.

        Args:
            data: DataFrame with price data
            strategy_func: Function(data, **params) -> (metrics, equity_curve)
            optimize_func: Function(data, param_grid) -> best_params
            param_grid: Dictionary of parameter ranges

        Returns:
            Analysis results
        """
        results = []
        split_size = len(data) // self.n_splits

        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            train_end = start_idx + int(split_size * self.train_ratio)

            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:end_idx]

            # Optimize on training data
            best_params = optimize_func(train_data, param_grid)

            # Backtest on training data (in-sample)
            is_metrics, is_equity = strategy_func(train_data, **best_params)

            # Backtest on test data (out-of-sample)
            oos_metrics, oos_equity = strategy_func(test_data, **best_params)

            results.append({
                'split': i + 1,
                'best_params': best_params,
                'in_sample': is_metrics,
                'out_of_sample': oos_metrics
            })

        # Analyze degradation
        degradation = self._analyze_degradation(results)

        return {
            'splits': results,
            'degradation': degradation,
            'is_overfit': degradation['sharpe_degradation'] > 0.4
        }

    def _analyze_degradation(self, results: List[Dict]) -> Dict:
        """Analyze performance degradation from IS to OOS."""
        is_sharpes = [r['in_sample']['sharpe_ratio'] for r in results]
        oos_sharpes = [r['out_of_sample']['sharpe_ratio'] for r in results]

        is_pf = [r['in_sample']['profit_factor'] for r in results]
        oos_pf = [r['out_of_sample']['profit_factor'] for r in results]

        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)

        sharpe_deg = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe if avg_is_sharpe != 0 else 0

        return {
            'avg_is_sharpe': avg_is_sharpe,
            'avg_oos_sharpe': avg_oos_sharpe,
            'sharpe_degradation': sharpe_deg,
            'avg_is_pf': np.mean(is_pf),
            'avg_oos_pf': np.mean(oos_pf)
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing.
    """

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate(self, trades: List[Dict]) -> Dict:
        """
        Run Monte Carlo simulations on trade sequence.

        Args:
            trades: List of trade dictionaries with 'pnl'

        Returns:
            Simulation results
        """
        pnls = np.array([t['pnl'] for t in trades])

        # Original metrics
        original_equity = self._build_equity_curve(pnls)
        original_dd = self._max_drawdown(original_equity)
        original_return = original_equity[-1] - 1

        # Simulated results
        sim_returns = []
        sim_drawdowns = []

        for _ in range(self.n_simulations):
            # Random reordering
            shuffled = np.random.permutation(pnls)
            equity = self._build_equity_curve(shuffled)
            sim_returns.append(equity[-1] - 1)
            sim_drawdowns.append(self._max_drawdown(equity))

        sim_returns = np.array(sim_returns)
        sim_drawdowns = np.array(sim_drawdowns)

        return {
            'original_return': original_return * 100,
            'original_max_dd': original_dd * 100,
            'simulated_return_mean': np.mean(sim_returns) * 100,
            'simulated_return_5th': np.percentile(sim_returns, 5) * 100,
            'simulated_return_95th': np.percentile(sim_returns, 95) * 100,
            'simulated_dd_mean': np.mean(sim_drawdowns) * 100,
            'simulated_dd_95th': np.percentile(sim_drawdowns, 95) * 100,
            'prob_profitable': np.mean(sim_returns > 0) * 100,
            'return_percentile': np.mean(sim_returns <= original_return) * 100
        }

    def _build_equity_curve(self, returns: np.ndarray, initial: float = 1.0) -> np.ndarray:
        """Build cumulative equity curve."""
        return initial * np.cumprod(1 + returns / 10000)

    def _max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return np.min(drawdown)


def validate_mt5_report(report_file: str) -> Dict:
    """
    Validate an MT5 Strategy Tester report.

    Args:
        report_file: Path to CSV or HTML report

    Returns:
        Validation results
    """
    # Parse report based on file type
    if report_file.endswith('.csv'):
        trades = _parse_csv_report(report_file)
    else:
        raise ValueError("Unsupported report format")

    # Build equity curve from trades
    equity_curve = []
    balance = 10000
    for trade in trades:
        balance += trade['pnl']
        equity_curve.append(balance)

    # Run validation
    validator = BacktestValidator()
    result = validator.validate(trades, equity_curve)

    # Run Monte Carlo
    mc = MonteCarloSimulator(n_simulations=5000)
    mc_results = mc.simulate(trades)

    result['monte_carlo'] = mc_results

    return result


def _parse_csv_report(filepath: str) -> List[Dict]:
    """Parse MT5 CSV trade report."""
    df = pd.read_csv(filepath)

    trades = []
    for _, row in df.iterrows():
        trades.append({
            'pnl': row.get('Profit', 0) + row.get('Commission', 0) + row.get('Swap', 0),
            'entry_time': row.get('OpenTime'),
            'exit_time': row.get('CloseTime'),
            'symbol': row.get('Symbol', ''),
            'type': row.get('Type', '')
        })

    return trades


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        report_file = sys.argv[1]
        results = validate_mt5_report(report_file)
        print(json.dumps(results, indent=2, default=str))
    else:
        print("Usage: python backtest_validator.py <report_file.csv>")
        print("\nExample with synthetic data:")

        # Generate synthetic trades
        np.random.seed(42)
        n_trades = 200
        win_rate = 0.55
        avg_win = 50
        avg_loss = -40

        trades = []
        for i in range(n_trades):
            if np.random.random() < win_rate:
                pnl = avg_win * (0.5 + np.random.random())
            else:
                pnl = avg_loss * (0.5 + np.random.random())

            trades.append({'pnl': pnl})

        # Build equity curve
        equity = []
        balance = 10000
        for t in trades:
            balance += t['pnl']
            equity.append(balance)

        # Validate
        validator = BacktestValidator()
        result = validator.validate(trades, equity)

        print("\nValidation Result:")
        print(f"  Valid: {result['valid']}")
        if result['issues']:
            print(f"  Issues: {result['issues']}")
        print(f"  Z-Score: {result['z_score']:.2f}")
        print("\nMetrics:")
        for k, v in result['metrics'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Monte Carlo
        mc = MonteCarloSimulator(n_simulations=5000)
        mc_results = mc.simulate(trades)

        print("\nMonte Carlo Results:")
        for k, v in mc_results.items():
            print(f"  {k}: {v:.2f}")
