"""
Performance reporting utilities for the Quantitative Trading System.
Provides standardized reporting of backtest results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .visualization import (
    plot_portfolio_performance, plot_return_distribution,
    plot_rolling_performance, plot_performance_table,
    plot_trade_analysis, plot_correlation_matrix
)

class PerformanceReport:
    """
    Generate comprehensive performance reports for trading strategies.
    """
    
    def __init__(self, strategy_name, output_dir='reports'):
        """
        Initialize the performance report.
        
        Parameters:
            strategy_name (str): Name of the strategy
            output_dir (str): Directory to save reports
        """
        self.strategy_name = strategy_name
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(output_dir, f"{strategy_name}_{self.timestamp}")
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
    
    def generate_report(self, backtest_results, metrics, trades=None, benchmark_name='Market'):
        """
        Generate a comprehensive performance report.
        
        Parameters:
            backtest_results (pd.DataFrame): Results from backtesting
            metrics (dict): Performance metrics
            trades (list, optional): List of trade dictionaries
            benchmark_name (str, optional): Name of the benchmark for comparison
            
        Returns:
            str: Path to the report directory
        """
        # Save summary metrics to CSV
        self._save_metrics_summary(metrics, benchmark_name)
        
        # Save all backtest results to CSV
        backtest_results.to_csv(os.path.join(self.report_dir, 'backtest_results.csv'))
        
        # Save trades to CSV if available
        if trades:
            self._save_trades_summary(trades)
        
        # Generate and save plots
        self._generate_plots(backtest_results, metrics, trades, benchmark_name)
        
        # Generate HTML report
        self._generate_html_report(backtest_results, metrics, trades, benchmark_name)
        
        return self.report_dir
    
    def _save_metrics_summary(self, metrics, benchmark_name):
        """Save performance metrics to CSV file"""
        # Create comparison dataframe
        summary = pd.DataFrame({
            'Metric': [
                'Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio',
                'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate',
                'Profit Factor', 'Alpha', 'Beta'
            ],
            'Strategy': [
                f"{metrics['total_return_strategy']:.2%}",
                f"{metrics['annual_return_strategy']:.2%}",
                f"{metrics['volatility_strategy']:.2%}",
                f"{metrics['sharpe_ratio_strategy']:.2f}",
                f"{metrics['sortino_ratio_strategy']:.2f}",
                f"{metrics['max_drawdown_strategy']:.2%}",
                f"{metrics['calmar_ratio_strategy']:.2f}",
                f"{metrics['win_rate']:.2%}" if 'win_rate' in metrics else 'N/A',
                f"{metrics['profit_factor']:.2f}" if 'profit_factor' in metrics else 'N/A',
                f"{metrics['alpha']:.2%}" if 'alpha' in metrics else 'N/A',
                f"{metrics['beta']:.2f}" if 'beta' in metrics else 'N/A'
            ],
            benchmark_name: [
                f"{metrics['total_return_market']:.2%}",
                f"{metrics['annual_return_market']:.2%}",
                f"{metrics['volatility_market']:.2%}",
                f"{metrics['sharpe_ratio_market']:.2f}",
                f"{metrics['sortino_ratio_market']:.2f}",
                f"{metrics['max_drawdown_market']:.2%}",
                f"{metrics['calmar_ratio_market']:.2f}",
                'N/A', 'N/A', 'N/A', 'N/A'
            ]
        })
        
        # Save to CSV
        summary.to_csv(os.path.join(self.report_dir, 'performance_metrics.csv'), index=False)
    
    def _save_trades_summary(self, trades):
        """Save trades to CSV file"""
        if not trades:
            return
        
        # Create trades dataframe
        trades_df = pd.DataFrame(trades)
        
        # Save to CSV
        trades_df.to_csv(os.path.join(self.report_dir, 'trades.csv'), index=False)
        
        # Calculate and save trade statistics
        stats = {
            'Total Trades': len(trades),
            'Winning Trades': sum(1 for t in trades if t['profit_loss'] > 0),
            'Losing Trades': sum(1 for t in trades if t['profit_loss'] <= 0),
            'Win Rate': sum(1 for t in trades if t['profit_loss'] > 0) / len(trades) if trades else 0,
            'Average Profit': sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / 
                             sum(1 for t in trades if t['profit_loss'] > 0) if sum(1 for t in trades if t['profit_loss'] > 0) > 0 else 0,
            'Average Loss': sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0) / 
                           sum(1 for t in trades if t['profit_loss'] <= 0) if sum(1 for t in trades if t['profit_loss'] <= 0) > 0 else 0,
            'Profit Factor': abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / 
                               sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0)) 
                               if sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0) != 0 else float('inf'),
            'Average Duration': sum(t['duration'] for t in trades) / len(trades) if trades else 0
        }
        
        # Save trade statistics
        pd.DataFrame([stats]).to_csv(os.path.join(self.report_dir, 'trade_statistics.csv'), index=False)
    
    def _generate_plots(self, backtest_results, metrics, trades, benchmark_name):
        """Generate and save visualization plots"""
        # Create plots directory
        plots_dir = os.path.join(self.report_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot portfolio performance
        fig_portfolio = plot_portfolio_performance(backtest_results, self.strategy_name)
        fig_portfolio.savefig(os.path.join(plots_dir, 'portfolio_performance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_portfolio)
        
        # Plot return distribution
        fig_returns = plot_return_distribution(backtest_results['Strategy_Net'], backtest_results['Return'])
        fig_returns.savefig(os.path.join(plots_dir, 'return_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_returns)
        
        # Plot rolling performance
        fig_rolling = plot_rolling_performance(backtest_results['Strategy_Net'], backtest_results['Return'])
        fig_rolling.savefig(os.path.join(plots_dir, 'rolling_performance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_rolling)
        
        # Plot performance metrics table
        fig_metrics = plot_performance_table(metrics)
        fig_metrics.savefig(os.path.join(plots_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_metrics)
        
        # Plot trade analysis if trades are available
        if trades and len(trades) > 0:
            fig_trades = plot_trade_analysis(trades)
            if fig_trades:
                fig_trades.savefig(os.path.join(plots_dir, 'trade_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_trades)
    
    def _generate_html_report(self, backtest_results, metrics, trades, benchmark_name):
        """Generate an HTML report"""
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.strategy_name} Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{self.strategy_name} Performance Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Strategy</th>
                        <th>{benchmark_name}</th>
                    </tr>
                    <tr>
                        <td>Total Return</td>
                        <td>{metrics['total_return_strategy']:.2%}</td>
                        <td>{metrics['total_return_market']:.2%}</td>
                    </tr>
                    <tr>
                        <td>Annual Return</td>
                        <td>{metrics['annual_return_strategy']:.2%}</td>
                        <td>{metrics['annual_return_market']:.2%}</td>
                    </tr>
                    <tr>
                        <td>Volatility</td>
                        <td>{metrics['volatility_strategy']:.2%}</td>
                        <td>{metrics['volatility_market']:.2%}</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{metrics['sharpe_ratio_strategy']:.2f}</td>
                        <td>{metrics['sharpe_ratio_market']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td>{metrics['sortino_ratio_strategy']:.2f}</td>
                        <td>{metrics['sortino_ratio_market']:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td>{metrics['max_drawdown_strategy']:.2%}</td>
                        <td>{metrics['max_drawdown_market']:.2%}</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{metrics['calmar_ratio_strategy']:.2f}</td>
                        <td>{metrics['calmar_ratio_market']:.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Portfolio Performance</h2>
                <div class="plot-container">
                    <img src="plots/portfolio_performance.png" alt="Portfolio Performance">
                </div>
            </div>
            
            <div class="section">
                <h2>Return Distribution</h2>
                <div class="plot-container">
                    <img src="plots/return_distribution.png" alt="Return Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Rolling Performance</h2>
                <div class="plot-container">
                    <img src="plots/rolling_performance.png" alt="Rolling Performance">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="plot-container">
                    <img src="plots/performance_metrics.png" alt="Performance Metrics">
                </div>
            </div>
        """
        
        # Add trade analysis section if trades are available
        if trades and len(trades) > 0:
            html_content += f"""
            <div class="section">
                <h2>Trade Analysis</h2>
                <div class="plot-container">
                    <img src="plots/trade_analysis.png" alt="Trade Analysis">
                </div>
                <h3>Trade Statistics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Trades</td>
                        <td>{len(trades)}</td>
                    </tr>
                    <tr>
                        <td>Winning Trades</td>
                        <td>{sum(1 for t in trades if t['profit_loss'] > 0)}</td>
                    </tr>
                    <tr>
                        <td>Losing Trades</td>
                        <td>{sum(1 for t in trades if t['profit_loss'] <= 0)}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{sum(1 for t in trades if t['profit_loss'] > 0) / len(trades):.2%}</td>
                    </tr>
                    <tr>
                        <td>Average Profit</td>
                        <td>{sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / sum(1 for t in trades if t['profit_loss'] > 0) if sum(1 for t in trades if t['profit_loss'] > 0) > 0 else 0:.2%}</td>
                    </tr>
                    <tr>
                        <td>Average Loss</td>
                        <td>{sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0) / sum(1 for t in trades if t['profit_loss'] <= 0) if sum(1 for t in trades if t['profit_loss'] <= 0) > 0 else 0:.2%}</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{abs(sum(t['profit_loss'] for t in trades if t['profit_loss'] > 0) / sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0)) if sum(t['profit_loss'] for t in trades if t['profit_loss'] <= 0) != 0 else float('inf'):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Duration (days)</td>
                        <td>{sum(t['duration'] for t in trades) / len(trades) if len(trades) > 0 else 0:.2f}</td>
                    </tr>
                </table>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(self.report_dir, 'report.html'), 'w') as f:
            f.write(html_content)

def generate_performance_report(strategy_name, backtest_results, metrics, trades=None, output_dir='reports'):
    """
    Generate a comprehensive performance report for a trading strategy.
    
    Parameters:
        strategy_name (str): Name of the strategy
        backtest_results (pd.DataFrame): Results from backtesting
        metrics (dict): Performance metrics
        trades (list, optional): List of trade dictionaries
        output_dir (str, optional): Directory to save reports
        
    Returns:
        str: Path to the report directory
    """
    report = PerformanceReport(strategy_name, output_dir)
    return report.generate_report(backtest_results, metrics, trades)

def compare_strategies(strategies_results, output_dir='reports'):
    """
    Generate a comparison report for multiple strategies.
    
    Parameters:
        strategies_results (dict): Dictionary mapping strategy names to 
                                  (backtest_results, metrics, trades) tuples
        output_dir (str, optional): Directory to save reports
        
    Returns:
        str: Path to the report directory
    """
    # Create report timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(output_dir, f"strategies_comparison_{timestamp}")
    
    # Create report directory
    os.makedirs(report_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(report_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metrics for all strategies
    comparison_data = {}
    for strategy_name, (_, metrics, _) in strategies_results.items():
        comparison_data[strategy_name] = {
            'Annual Return': metrics['annual_return_strategy'],
            'Volatility': metrics['volatility_strategy'],
            'Sharpe Ratio': metrics['sharpe_ratio_strategy'],
            'Max Drawdown': metrics['max_drawdown_strategy'],
            'Win Rate': metrics.get('win_rate', 0),
            'Profit Factor': metrics.get('profit_factor', 0)
        }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(report_dir, 'strategies_comparison.csv'))
    
    # Plot comparison charts
    # 1. Bar charts for key metrics
    key_metrics = ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        comparison_df.loc[metric].plot(kind='bar', ax=axes[i])
        axes[i].set_title(metric)
        axes[i].grid(True)
        
        # Format percentage metrics
        if metric in ['Annual Return', 'Max Drawdown', 'Win Rate']:
            from matplotlib.ticker import FuncFormatter
            axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'key_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Plot equity curves for all strategies
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy_name, (backtest_results, _, _) in strategies_results.items():
        ax.plot(backtest_results.index, backtest_results['Cumulative_Strategy'], 
                label=strategy_name)
    
    # Also plot market
    for strategy_name, (backtest_results, _, _) in strategies_results.items():
        ax.plot(backtest_results.index, backtest_results['Cumulative_Market'], 
                'k--', label='Market')
        break  # Only need one market curve
    
    ax.set_title('Equity Curves Comparison')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    
    fig.savefig(os.path.join(plots_dir, 'equity_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Plot correlation matrix
    strategies_returns = {}
    for strategy_name, (backtest_results, _, _) in strategies_results.items():
        strategies_returns[strategy_name] = backtest_results['Strategy_Net']
    
    # Add market returns
    for strategy_name, (backtest_results, _, _) in strategies_results.items():
        strategies_returns['Market'] = backtest_results['Return']
        break
    
    # Create correlation matrix
    fig = plot_correlation_matrix(strategies_returns)
    fig.savefig(os.path.join(plots_dir, 'returns_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Strategies Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .plot-container {{ margin: 20px 0; text-align: center; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Strategies Comparison Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Performance Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    {' '.join(f'<th>{strategy}</th>' for strategy in comparison_data.keys())}
                </tr>
    """
    
    # Add rows for each metric
    for metric in comparison_df.index:
        html_content += f"<tr><td>{metric}</td>"
        
        for strategy in comparison_data.keys():
            value = comparison_df.loc[metric, strategy]
            formatted_value = f"{value:.2%}" if metric in ['Annual Return', 'Volatility', 'Max Drawdown', 'Win Rate'] else f"{value:.2f}"
            html_content += f"<td>{formatted_value}</td>"
        
        html_content += "</tr>"
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Key Metrics Comparison</h2>
            <div class="plot-container">
                <img src="plots/key_metrics_comparison.png" alt="Key Metrics Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Equity Curves Comparison</h2>
            <div class="plot-container">
                <img src="plots/equity_curves_comparison.png" alt="Equity Curves Comparison">
            </div>
        </div>
        
        <div class="section">
            <h2>Returns Correlation</h2>
            <div class="plot-container">
                <img src="plots/returns_correlation.png" alt="Returns Correlation">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(report_dir, 'strategies_comparison.html'), 'w') as f:
        f.write(html_content)
    
    return report_dir