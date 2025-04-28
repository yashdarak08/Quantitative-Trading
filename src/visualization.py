import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import datetime, timedelta

def plot_portfolio_performance(backtest_results, strategy_name=None, figsize=(12, 10)):
    """
    Plot detailed performance charts for the trading strategy.
    
    Parameters:
        backtest_results (pd.DataFrame): DataFrame with backtest results
        strategy_name (str, optional): Name of the strategy
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Set up the plots
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax3 = plt.subplot2grid((4, 1), (3, 0))
    
    # Plot equity curves
    ax1.plot(backtest_results.index, backtest_results['Cumulative_Market'], 'b-', label='Market')
    ax1.plot(backtest_results.index, backtest_results['Cumulative_Strategy'], 'g-', label='Strategy')
    title = 'Strategy vs Market Performance'
    if strategy_name:
        title = f'{strategy_name}: {title}'
    ax1.set_title(title)
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Calculate drawdowns
    market_dd = 1 - backtest_results['Cumulative_Market'] / backtest_results['Cumulative_Market'].cummax()
    strategy_dd = 1 - backtest_results['Cumulative_Strategy'] / backtest_results['Cumulative_Strategy'].cummax()
    
    # Plot drawdowns
    ax2.fill_between(backtest_results.index, 0, -100*market_dd, color='blue', alpha=0.2)
    ax2.fill_between(backtest_results.index, 0, -100*strategy_dd, color='green', alpha=0.3)
    ax2.plot(backtest_results.index, -100*market_dd, 'b-', label='Market')
    ax2.plot(backtest_results.index, -100*strategy_dd, 'g-', label='Strategy')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot positions or signals
    if 'Position' in backtest_results.columns:
        ax3.fill_between(backtest_results.index, 0, 100*backtest_results['Position'], where=backtest_results['Position']>0, color='green', alpha=0.3)
        ax3.fill_between(backtest_results.index, 0, 100*backtest_results['Position'], where=backtest_results['Position']<0, color='red', alpha=0.3)
        ax3.plot(backtest_results.index, 100*backtest_results['Position'], 'k-', label='Position Size (%)')
    else:
        ax3.plot(backtest_results.index, backtest_results['Signal'], 'k-', label='Signal')
    ax3.set_ylabel('Position Size (%)')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    # Format the date axis
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_return_distribution(returns, benchmark_returns=None, figsize=(10, 6)):
    """
    Plot return distribution histogram and statistics.
    
    Parameters:
        returns (pd.Series): Strategy returns
        benchmark_returns (pd.Series, optional): Benchmark returns for comparison
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Calculate return statistics
    mean_return = returns.mean()
    median_return = returns.median()
    std_return = returns.std()
    skew_return = returns.skew()
    kurt_return = returns.kurtosis()
    
    # Plot histogram of returns
    plt.hist(returns, bins=50, alpha=0.5, color='green', label='Strategy')
    
    if benchmark_returns is not None:
        plt.hist(benchmark_returns, bins=50, alpha=0.5, color='blue', label='Benchmark')
    
    # Add vertical lines for mean and median
    plt.axvline(mean_return, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_return:.4%}')
    plt.axvline(median_return, color='black', linestyle='dashed', linewidth=1, label=f'Median: {median_return:.4%}')
    
    # Add normal distribution overlay
    x = np.linspace(min(returns), max(returns), 100)
    plt.plot(x, len(returns) * (1 / (std_return * np.sqrt(2 * np.pi))) * 
             np.exp(-(x - mean_return) ** 2 / (2 * std_return ** 2)) * (max(returns) - min(returns)) / 50, 
             'r-', linewidth=1, label='Normal Distribution')
    
    # Add statistics text box
    stats_text = (f'Mean: {mean_return:.4%}\n'
                 f'Median: {median_return:.4%}\n'
                 f'Std Dev: {std_return:.4%}\n'
                 f'Skewness: {skew_return:.2f}\n'
                 f'Kurtosis: {kurt_return:.2f}\n'
                 f'Min: {returns.min():.4%}\n'
                 f'Max: {returns.max():.4%}')
    
    # Position the text box
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis as percentage
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    plt.tight_layout()
    return fig

def plot_rolling_performance(returns, benchmark_returns=None, window=252, figsize=(12, 12)):
    """
    Plot rolling performance metrics.
    
    Parameters:
        returns (pd.Series): Strategy returns
        benchmark_returns (pd.Series, optional): Benchmark returns for comparison
        window (int, optional): Window size for rolling metrics
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Calculate rolling metrics
    rolling_return = returns.rolling(window=window).mean() * window
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Setup subplots
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    
    # Plot rolling annualized return
    ax1.plot(rolling_return.index, rolling_return, 'g-', label='Strategy')
    if benchmark_returns is not None:
        benchmark_rolling_return = benchmark_returns.rolling(window=window).mean() * window
        ax1.plot(benchmark_rolling_return.index, benchmark_rolling_return, 'b-', label='Benchmark')
    ax1.set_title(f'Rolling {window}-day Annualized Return')
    ax1.set_ylabel('Return')
    ax1.legend()
    ax1.grid(True)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Plot rolling volatility
    ax2.plot(rolling_vol.index, rolling_vol, 'g-', label='Strategy')
    if benchmark_returns is not None:
        benchmark_rolling_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(window)
        ax2.plot(benchmark_rolling_vol.index, benchmark_rolling_vol, 'b-', label='Benchmark')
    ax2.set_title(f'Rolling {window}-day Annualized Volatility')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Plot rolling Sharpe ratio
    ax3.plot(rolling_sharpe.index, rolling_sharpe, 'g-', label='Strategy')
    if benchmark_returns is not None:
        benchmark_rolling_sharpe = benchmark_rolling_return / benchmark_rolling_vol
        ax3.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe, 'b-', label='Benchmark')
    ax3.set_title(f'Rolling {window}-day Sharpe Ratio')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    # Format the date axis
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_performance_table(metrics, figsize=(10, 6)):
    """
    Create a visual table of performance metrics.
    
    Parameters:
        metrics (dict): Dictionary of performance metrics
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create data for table
    data = []
    headers = ['Metric', 'Strategy', 'Market']
    
    # Add returns metrics
    data.append(['Total Return', f"{metrics['total_return_strategy']:.2%}", f"{metrics['total_return_market']:.2%}"])
    data.append(['Annual Return', f"{metrics['annual_return_strategy']:.2%}", f"{metrics['annual_return_market']:.2%}"])
    data.append(['Volatility', f"{metrics['volatility_strategy']:.2%}", f"{metrics['volatility_market']:.2%}"])
    data.append(['Max Drawdown', f"{metrics['max_drawdown_strategy']:.2%}", f"{metrics['max_drawdown_market']:.2%}"])
    
    # Add risk-adjusted metrics
    data.append(['Sharpe Ratio', f"{metrics['sharpe_ratio_strategy']:.2f}", f"{metrics['sharpe_ratio_market']:.2f}"])
    data.append(['Sortino Ratio', f"{metrics['sortino_ratio_strategy']:.2f}", f"{metrics['sortino_ratio_market']:.2f}"])
    data.append(['Calmar Ratio', f"{metrics['calmar_ratio_strategy']:.2f}", f"{metrics['calmar_ratio_market']:.2f}"])
    
    # Add other metrics
    data.append(['Alpha', f"{metrics['alpha']:.2%}", ""])
    data.append(['Beta', f"{metrics['beta']:.2f}", ""])
    data.append(['Information Ratio', f"{metrics['information_ratio']:.2f}", ""])
    
    # Add trade statistics
    data.append(['Win Rate', f"{metrics['win_rate']:.2%}", ""])
    data.append(['Profit Factor', f"{metrics['profit_factor']:.2f}", ""])
    data.append(['Total Trades', f"{metrics['total_trades']}", ""])
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc='center',
                    cellLoc='center', colColours=['#f2f2f2']*3)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code performance comparison
    for i in range(len(data)):
        # Skip cells without comparison
        if i >= 7 and i <= 12:
            continue
        
        # Get values for comparison (remove % and convert to float)
        try:
            strategy_val = float(data[i][1].strip('%'))
            market_val = float(data[i][2].strip('%'))
            
            # Set colors based on better performance (higher is better except for volatility and drawdown)
            if i in [2, 3]:  # Volatility and drawdown (lower is better)
                if strategy_val < market_val:
                    table[(i+1, 1)].set_facecolor('#c6efce')  # Light green
                    table[(i+1, 2)].set_facecolor('#ffc7ce')  # Light red
                elif strategy_val > market_val:
                    table[(i+1, 1)].set_facecolor('#ffc7ce')
                    table[(i+1, 2)].set_facecolor('#c6efce')
            else:  # All other metrics (higher is better)
                if strategy_val > market_val:
                    table[(i+1, 1)].set_facecolor('#c6efce')
                    table[(i+1, 2)].set_facecolor('#ffc7ce')
                elif strategy_val < market_val:
                    table[(i+1, 1)].set_facecolor('#ffc7ce')
                    table[(i+1, 2)].set_facecolor('#c6efce')
        except:
            pass
    
    plt.title('Performance Metrics', fontsize=14)
    plt.tight_layout()
    
    return fig

def plot_trade_analysis(trades, figsize=(14, 10)):
    """
    Plot trade analysis charts.
    
    Parameters:
        trades (list): List of trade dictionaries
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    if not trades:
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # Extract trade data
    profits = [t['profit_loss'] for t in trades]
    durations = [t['duration'] for t in trades]
    dates = [t['entry_date'] for t in trades]
    
    # Calculate trade statistics
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    avg_profit = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
    avg_loss = np.mean([p for p in profits if p <= 0]) if any(p <= 0 for p in profits) else 0
    
    # Setup subplots
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    
    # Plot profit distribution
    ax1.hist(profits, bins=20, alpha=0.7, color='green')
    ax1.axvline(0, color='red', linestyle='--')
    ax1.set_title('Trade Profit/Loss Distribution')
    ax1.set_xlabel('Profit/Loss (%)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True)
    
    # Plot profit by trade
    colors = ['green' if p > 0 else 'red' for p in profits]
    ax2.bar(range(len(profits)), profits, color=colors, alpha=0.7)
    ax2.set_title('Profit/Loss by Trade')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Profit/Loss (%)')
    ax2.grid(True)
    
    # Plot trade duration distribution
    ax3.hist(durations, bins=20, alpha=0.7, color='blue')
    ax3.set_title('Trade Duration Distribution')
    ax3.set_xlabel('Duration (days)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    # Plot trade profit vs duration scatter
    ax4.scatter(durations, profits, alpha=0.7, c=colors)
    ax4.set_title('Profit/Loss vs Duration')
    ax4.set_xlabel('Duration (days)')
    ax4.set_ylabel('Profit/Loss (%)')
    ax4.grid(True)
    
    # Add summary statistics
    stats_text = (f'Total Trades: {len(trades)}\n'
                 f'Win Rate: {win_rate:.2%}\n'
                 f'Avg Profit: {avg_profit:.2%}\n'
                 f'Avg Loss: {avg_loss:.2%}\n'
                 f'Profit Factor: {-np.sum([p for p in profits if p > 0]) / np.sum([p for p in profits if p <= 0]):.2f}' if sum(p < 0 for p in profits) > 0 else 'Profit Factor: ∞')
    
    # Position the text box in the first subplot
    ax1.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_performance_report(backtest_results, metrics, trades, strategy_name=None):
    """
    Create a comprehensive performance report with multiple visualizations.
    
    Parameters:
        backtest_results (pd.DataFrame): DataFrame with backtest results
        metrics (dict): Dictionary of performance metrics
        trades (list): List of trade dictionaries
        strategy_name (str, optional): Name of the strategy
        
    Returns:
        list: List of matplotlib figures
    """
    figures = []
    
    # 1. Portfolio Performance
    fig1 = plot_portfolio_performance(backtest_results, strategy_name)
    figures.append(fig1)
    
    # 2. Return Distribution
    fig2 = plot_return_distribution(backtest_results['Strategy_Net'], backtest_results['Return'])
    figures.append(fig2)
    
    # 3. Rolling Performance
    fig3 = plot_rolling_performance(backtest_results['Strategy_Net'], backtest_results['Return'])
    figures.append(fig3)
    
    # 4. Performance Metrics Table
    fig4 = plot_performance_table(metrics)
    figures.append(fig4)
    
    # 5. Trade Analysis (if trades are available)
    if trades:
        fig5 = plot_trade_analysis(trades)
        figures.append(fig5)
    
    return figures

def plot_indicator_analysis(price_data, indicator_data, strategy_signals=None, window=None, figsize=(14, 10)):
    """
    Plot price with technical indicators and strategy signals.
    
    Parameters:
        price_data (pd.Series): Price series
        indicator_data (dict): Dictionary of indicator series
        strategy_signals (pd.Series, optional): Strategy signals
        window (int, optional): Number of periods to display (most recent)
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # If window is specified, limit the data
    if window and window < len(price_data):
        price_data = price_data.iloc[-window:]
        for key in indicator_data:
            indicator_data[key] = indicator_data[key].iloc[-window:]
        if strategy_signals is not None:
            strategy_signals = strategy_signals.iloc[-window:]
    
    # Set up the plots
    if strategy_signals is not None:
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((5, 1), (3, 0))
        ax3 = plt.subplot2grid((5, 1), (4, 0))
    else:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0))
        ax3 = None
    
    # Plot price
    ax1.plot(price_data.index, price_data, 'b-', label='Price')
    ax1.set_title('Price with Indicators')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Add indicators to the price chart
    colors = ['g', 'r', 'c', 'm', 'y', 'k']
    color_idx = 0
    
    # Create a second y-axis for normalized indicators
    ax1b = ax1.twinx()
    
    for name, indicator in indicator_data.items():
        # Determine if indicator should be on primary or secondary axis
        if 'MA' in name or 'Average' in name or 'Band' in name:
            # Price-scale indicators go on primary axis
            ax1.plot(indicator.index, indicator, colors[color_idx % len(colors)]+'-', label=name)
            ax1.legend(loc='upper left')
        else:
            # Normalized indicators go on secondary axis
            ax1b.plot(indicator.index, indicator, colors[color_idx % len(colors)]+'-', label=name)
            ax1b.legend(loc='upper right')
        
        color_idx += 1
    
    # Plot volume if available
    if 'Volume' in price_data.columns:
        ax2.bar(price_data.index, price_data['Volume'], color='blue', alpha=0.5)
        ax2.set_title('Volume')
        ax2.grid(True)
    else:
        # If no volume, use one of the indicators for the secondary chart
        for name, indicator in indicator_data.items():
            if 'RSI' in name or 'Oscillator' in name or 'Index' in name:
                ax2.plot(indicator.index, indicator, 'g-', label=name)
                ax2.set_title(name)
                ax2.grid(True)
                ax2.axhline(y=70, color='r', linestyle='--')  # Overbought
                ax2.axhline(y=30, color='g', linestyle='--')  # Oversold
                break
    
    # Plot signals if provided
    if strategy_signals is not None and ax3 is not None:
        ax3.fill_between(strategy_signals.index, 0, strategy_signals, where=strategy_signals>0, color='green', alpha=0.3)
        ax3.fill_between(strategy_signals.index, 0, strategy_signals, where=strategy_signals<0, color='red', alpha=0.3)
        ax3.plot(strategy_signals.index, strategy_signals, 'k-', label='Signal')
        ax3.set_title('Strategy Signals')
        ax3.set_ylabel('Signal')
        ax3.set_ylim(-1.5, 1.5)  # Adjust limits for signal visualization
        ax3.grid(True)
    
    # Format the date axis
    for ax in [ax1, ax2]:
        if ax:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if ax3:
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(returns_dict, figsize=(10, 8)):
    """
    Plot correlation matrix heatmap for multiple return series.
    
    Parameters:
        returns_dict (dict): Dictionary of return series
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    # Create DataFrame with all return series
    returns_df = pd.DataFrame({k: v for k, v in returns_dict.items()})
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Return Correlation Matrix')
    plt.tight_layout()
    
    return fig

def plot_regime_analysis(returns, volatility, figsize=(14, 7)):
    """
    Plot regime analysis based on return and volatility.
    
    Parameters:
        returns (pd.Series): Return series
        volatility (pd.Series): Volatility series
        figsize (tuple, optional): Figure size
        
    Returns:
        fig: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate z-scores for returns and volatility
    returns_z = (returns - returns.mean()) / returns.std()
    volatility_z = (volatility - volatility.mean()) / volatility.std()
    
    # Define regimes
    bull_high_vol = (returns_z > 0) & (volatility_z > 0)
    bull_low_vol = (returns_z > 0) & (volatility_z <= 0)
    bear_high_vol = (returns_z <= 0) & (volatility_z > 0)
    bear_low_vol = (returns_z <= 0) & (volatility_z <= 0)
    
    # Plot scatter with regimes
    ax.scatter(volatility[bull_high_vol], returns[bull_high_vol], 
              color='green', alpha=0.5, label='Bull Market / High Volatility')
    ax.scatter(volatility[bull_low_vol], returns[bull_low_vol], 
              color='blue', alpha=0.5, label='Bull Market / Low Volatility')
    ax.scatter(volatility[bear_high_vol], returns[bear_high_vol], 
              color='red', alpha=0.5, label='Bear Market / High Volatility')
    ax.scatter(volatility[bear_low_vol], returns[bear_low_vol], 
              color='orange', alpha=0.5, label='Bear Market / Low Volatility')
    
    # Add horizontal and vertical lines at means
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axvline(x=volatility.mean(), color='k', linestyle='--')
    
    # Add labels and title
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_title('Market Regimes')
    ax.legend()
    ax.grid(True)
    
    return fig

def save_plots(figures, directory='reports', prefix='', timestamp=None):
    """
    Save a list of matplotlib figures to files.
    
    Parameters:
        figures (list): List of figure objects
        directory (str, optional): Directory to save plots
        prefix (str, optional): Prefix for filenames
        timestamp (str, optional): Timestamp string for filenames
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save each figure
    for i, fig in enumerate(figures):
        if fig is not None:
            filename = os.path.join(directory, f"{prefix}_{timestamp}_{i}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved {len([f for f in figures if f is not None])} plots to {directory}/")

def create_comprehensive_report(backtest_results, metrics, trades, strategies_comparison=None, save_path=None):
    """
    Create a comprehensive performance report with all visualizations.
    
    Parameters:
        backtest_results (pd.DataFrame): DataFrame with backtest results
        metrics (dict): Dictionary of performance metrics
        trades (list): List of trade dictionaries
        strategies_comparison (dict, optional): Dict of strategy names to their metrics for comparison
        save_path (str, optional): Directory to save the report figures
        
    Returns:
        dict: Dictionary of figure objects
    """
    figures = {}
    
    # Portfolio performance
    figures['portfolio'] = plot_portfolio_performance(backtest_results, 
                                                     strategy_name=metrics.get('strategy_name', 'Strategy'))
    
    # Return distribution
    figures['returns'] = plot_return_distribution(backtest_results['Strategy_Net'], 
                                                 benchmark_returns=backtest_results['Return'])
    
    # Rolling performance
    figures['rolling'] = plot_rolling_performance(backtest_results['Strategy_Net'], 
                                                 benchmark_returns=backtest_results['Return'])
    
    # Performance metrics table
    figures['metrics'] = plot_performance_table(metrics)
    
    # Trade analysis
    if trades and len(trades) > 0:
        figures['trades'] = plot_trade_analysis(trades)
    
    # Strategies comparison
    if strategies_comparison:
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            name: {
                'Sharpe Ratio': metrics['sharpe_ratio_strategy'],
                'Annual Return': metrics['annual_return_strategy'],
                'Max Drawdown': metrics['max_drawdown_strategy'],
                'Win Rate': metrics.get('win_rate', 0),
            } for name, metrics in strategies_comparison.items()
        })
        
        # Plot comparison bar charts
        figures['strategies_comparison'] = plt.figure(figsize=(12, 10))
        for i, metric in enumerate(['Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'Win Rate']):
            ax = plt.subplot(2, 2, i+1)
            comparison_df.loc[metric].plot(kind='bar', ax=ax)
            ax.set_title(metric)
            ax.grid(True)
            
            # Format percentage metrics
            if metric in ['Annual Return', 'Max Drawdown', 'Win Rate']:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
    
    # Save figures if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for name, fig in figures.items():
            if fig:
                fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches='tight')
    
    return figures