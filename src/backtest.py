import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class BacktestEngine:
    """
    Enhanced backtesting engine with realistic trading simulation and performance metrics.
    """
    
    def __init__(self, data, strategy, config=None):
        """
        Initialize the backtesting engine.
        
        Parameters:
            data (pd.DataFrame): DataFrame with price data
            strategy (Strategy): Trading strategy object
            config (dict, optional): Configuration parameters
        """
        self.data = data.copy()
        self.strategy = strategy
        self.config = config or {}
        
        # Initialize default parameters from config
        self.transaction_cost = self.config.get('transaction_cost', 0.001)  # 0.1% transaction cost
        self.slippage = self.config.get('slippage', 0.001)  # 0.1% slippage
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual risk-free rate
        self.rebalance_frequency = self.config.get('rebalance_frequency', '1D')  # Daily rebalancing
        
        # Initialize results
        self.positions = None
        self.portfolio = None
        self.trades = []
        self.performance_metrics = {}
    
    def run(self):
        """
        Run the backtest.
        
        Returns:
            pd.DataFrame: DataFrame with portfolio values and metrics
        """
        # Generate trading signals
        signals = self.strategy.generate_signals(self.data)
        
        # Apply risk management
        risk_params = {
            'stop_loss': self.config.get('stop_loss', 0.05),
            'take_profit': self.config.get('take_profit', 0.1),
            'max_position_size': self.config.get('max_position_size', 0.2),
            'min_position_size': self.config.get('min_position_size', 0.05),
            'max_drawdown': self.config.get('max_drawdown', 0.2)
        }
        self.positions = self.strategy.apply_risk_management(signals, self.data, risk_params)
        
        # Initialize portfolio dataframe
        self.portfolio = self.data.copy()
        self.portfolio['Signal'] = self.positions
        self.portfolio['Position'] = self.positions.shift(1).fillna(0)  # Apply signal from previous day
        
        # Calculate returns
        self.portfolio['Return'] = self.portfolio['Price'].pct_change()
        
        # Apply slippage and transaction costs
        self.portfolio['Transaction'] = abs(self.portfolio['Position'].diff())
        self.portfolio['Slippage'] = self.portfolio['Transaction'] * self.slippage
        self.portfolio['Cost'] = self.portfolio['Transaction'] * self.transaction_cost
        
        # Calculate strategy returns with costs
        self.portfolio['Strategy_Gross'] = self.portfolio['Position'] * self.portfolio['Return']
        self.portfolio['Strategy_Net'] = self.portfolio['Strategy_Gross'] - self.portfolio['Slippage'] - self.portfolio['Cost']
        
        # Calculate cumulative returns
        self.portfolio['Cumulative_Market'] = (1 + self.portfolio['Return']).cumprod()
        self.portfolio['Cumulative_Strategy'] = (1 + self.portfolio['Strategy_Net']).cumprod()
        
        # Record trades
        self._record_trades()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.portfolio
        
    def _record_trades(self):
        """Record trades for analysis"""
        position_changes = self.portfolio['Position'].diff()
        
        # Find entry and exit points
        for i in range(1, len(position_changes)):
            if position_changes.iloc[i] != 0:
                # Record the trade
                trade = {
                    'entry_date': self.portfolio.index[i],
                    'entry_price': self.portfolio['Price'].iloc[i],
                    'position': position_changes.iloc[i],
                    'exit_date': None,
                    'exit_price': None,
                    'profit_loss': None,
                    'duration': None
                }
                
                # Find the exit point
                for j in range(i+1, len(position_changes)):
                    if (position_changes.iloc[j] != 0 and 
                        np.sign(position_changes.iloc[j]) != np.sign(position_changes.iloc[i])):
                        trade['exit_date'] = self.portfolio.index[j]
                        trade['exit_price'] = self.portfolio['Price'].iloc[j]
                        
                        # Calculate profit/loss
                        if trade['position'] > 0:  # Long position
                            trade['profit_loss'] = (trade['exit_price'] / trade['entry_price'] - 1) * 100
                        else:  # Short position
                            trade['profit_loss'] = (trade['entry_price'] / trade['exit_price'] - 1) * 100
                        
                        # Calculate duration
                        trade['duration'] = (trade['exit_date'] - trade['entry_date']).days
                        
                        break
                
                # If no exit found, consider the last day as exit
                if trade['exit_date'] is None:
                    trade['exit_date'] = self.portfolio.index[-1]
                    trade['exit_price'] = self.portfolio['Price'].iloc[-1]
                    
                    # Calculate profit/loss
                    if trade['position'] > 0:  # Long position
                        trade['profit_loss'] = (trade['exit_price'] / trade['entry_price'] - 1) * 100
                    else:  # Short position
                        trade['profit_loss'] = (trade['entry_price'] / trade['exit_price'] - 1) * 100
                    
                    # Calculate duration
                    trade['duration'] = (trade['exit_date'] - trade['entry_date']).days
                
                self.trades.append(trade)
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        # Extract returns
        strategy_returns = self.portfolio['Strategy_Net'].dropna()
        market_returns = self.portfolio['Return'].dropna()
        
        # Annualization factor based on data frequency
        if len(strategy_returns) > 252:  # Daily data
            annualization_factor = 252
        elif len(strategy_returns) > 52:  # Weekly data
            annualization_factor = 52
        else:  # Monthly data
            annualization_factor = 12
        
        # 1. Total Return
        total_return_strategy = self.portfolio['Cumulative_Strategy'].iloc[-1] - 1
        total_return_market = self.portfolio['Cumulative_Market'].iloc[-1] - 1
        
        # 2. Annualized Return
        years = len(strategy_returns) / annualization_factor
        annual_return_strategy = (1 + total_return_strategy) ** (1 / years) - 1
        annual_return_market = (1 + total_return_market) ** (1 / years) - 1
        
        # 3. Volatility
        volatility_strategy = strategy_returns.std() * np.sqrt(annualization_factor)
        volatility_market = market_returns.std() * np.sqrt(annualization_factor)
        
        # 4. Sharpe Ratio
        excess_return_strategy = annual_return_strategy - self.risk_free_rate
        excess_return_market = annual_return_market - self.risk_free_rate
        sharpe_ratio_strategy = excess_return_strategy / volatility_strategy if volatility_strategy != 0 else 0
        sharpe_ratio_market = excess_return_market / volatility_market if volatility_market != 0 else 0
        
        # 5. Sortino Ratio (downside risk only)
        downside_returns_strategy = strategy_returns[strategy_returns < 0]
        downside_returns_market = market_returns[market_returns < 0]
        downside_volatility_strategy = downside_returns_strategy.std() * np.sqrt(annualization_factor)
        downside_volatility_market = downside_returns_market.std() * np.sqrt(annualization_factor)
        sortino_ratio_strategy = excess_return_strategy / downside_volatility_strategy if downside_volatility_strategy != 0 else 0
        sortino_ratio_market = excess_return_market / downside_volatility_market if downside_volatility_market != 0 else 0
        
        # 6. Maximum Drawdown
        cumulative_strategy = self.portfolio['Cumulative_Strategy']
        cumulative_market = self.portfolio['Cumulative_Market']
        
        # Calculate running maximum
        running_max_strategy = cumulative_strategy.cummax()
        running_max_market = cumulative_market.cummax()
        
        # Calculate drawdown
        drawdown_strategy = (cumulative_strategy / running_max_strategy - 1)
        drawdown_market = (cumulative_market / running_max_market - 1)
        
        max_drawdown_strategy = drawdown_strategy.min()
        max_drawdown_market = drawdown_market.min()
        
        # 7. Calmar Ratio
        calmar_ratio_strategy = annual_return_strategy / abs(max_drawdown_strategy) if max_drawdown_strategy != 0 else 0
        calmar_ratio_market = annual_return_market / abs(max_drawdown_market) if max_drawdown_market != 0 else 0
        
        # 8. Win Rate
        winning_trades = len([trade for trade in self.trades if trade['profit_loss'] > 0])
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 9. Average Profit/Loss
        avg_profit = np.mean([trade['profit_loss'] for trade in self.trades if trade['profit_loss'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade['profit_loss'] for trade in self.trades if trade['profit_loss'] <= 0]) if total_trades - winning_trades > 0 else 0
        
        # 10. Profit Factor
        total_profit = sum([trade['profit_loss'] for trade in self.trades if trade['profit_loss'] > 0])
        total_loss = abs(sum([trade['profit_loss'] for trade in self.trades if trade['profit_loss'] <= 0]))
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
        
        # 11. Average Trade Duration
        avg_duration = np.mean([trade['duration'] for trade in self.trades]) if total_trades > 0 else 0
        
        # 12. Beta and Alpha
        if not market_returns.empty and not strategy_returns.empty:
            # Calculate covariance and variance
            covariance = strategy_returns.cov(market_returns)
            variance = market_returns.var()
            
            # Calculate beta
            beta = covariance / variance if variance != 0 else 0
            
            # Calculate alpha
            alpha = annual_return_strategy - (self.risk_free_rate + beta * (annual_return_market - self.risk_free_rate))
        else:
            beta = 0
            alpha = 0
        
        # 13. Information Ratio
        tracking_error = (strategy_returns - market_returns).std() * np.sqrt(annualization_factor)
        information_ratio = (annual_return_strategy - annual_return_market) / tracking_error if tracking_error != 0 else 0
        
        # Store all metrics
        self.performance_metrics = {
            'total_return_strategy': total_return_strategy,
            'total_return_market': total_return_market,
            'annual_return_strategy': annual_return_strategy,
            'annual_return_market': annual_return_market,
            'volatility_strategy': volatility_strategy,
            'volatility_market': volatility_market,
            'sharpe_ratio_strategy': sharpe_ratio_strategy,
            'sharpe_ratio_market': sharpe_ratio_market,
            'sortino_ratio_strategy': sortino_ratio_strategy,
            'sortino_ratio_market': sortino_ratio_market,
            'max_drawdown_strategy': max_drawdown_strategy,
            'max_drawdown_market': max_drawdown_market,
            'calmar_ratio_strategy': calmar_ratio_strategy,
            'calmar_ratio_market': calmar_ratio_market,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'total_trades': total_trades
        }
    
    def get_performance_summary(self):
        """Get a formatted performance summary"""
        if not self.performance_metrics:
            return "Backtest not yet run."
        
        metrics = self.performance_metrics
        
        summary = f"""
        ============= PERFORMANCE SUMMARY =============
        Strategy: {self.strategy.name}
        Period: {self.portfolio.index[0].strftime('%Y-%m-%d')} to {self.portfolio.index[-1].strftime('%Y-%m-%d')}
        
        RETURNS:
        Total Return (Strategy): {metrics['total_return_strategy']:.2%}
        Total Return (Market): {metrics['total_return_market']:.2%}
        Annualized Return (Strategy): {metrics['annual_return_strategy']:.2%}
        Annualized Return (Market): {metrics['annual_return_market']:.2%}
        Alpha: {metrics['alpha']:.2%}
        Beta: {metrics['beta']:.2f}
        
        RISK METRICS:
        Volatility (Strategy): {metrics['volatility_strategy']:.2%}
        Volatility (Market): {metrics['volatility_market']:.2%}
        Maximum Drawdown (Strategy): {metrics['max_drawdown_strategy']:.2%}
        Maximum Drawdown (Market): {metrics['max_drawdown_market']:.2%}
        
        RISK-ADJUSTED PERFORMANCE:
        Sharpe Ratio (Strategy): {metrics['sharpe_ratio_strategy']:.2f}
        Sharpe Ratio (Market): {metrics['sharpe_ratio_market']:.2f}
        Sortino Ratio (Strategy): {metrics['sortino_ratio_strategy']:.2f}
        Sortino Ratio (Market): {metrics['sortino_ratio_market']:.2f}
        Calmar Ratio (Strategy): {metrics['calmar_ratio_strategy']:.2f}
        Calmar Ratio (Market): {metrics['calmar_ratio_market']:.2f}
        Information Ratio: {metrics['information_ratio']:.2f}
        
        TRADE STATISTICS:
        Total Trades: {metrics['total_trades']}
        Win Rate: {metrics['win_rate']:.2%}
        Profit Factor: {metrics['profit_factor']:.2f}
        Average Profit: {metrics['avg_profit']:.2%}
        Average Loss: {metrics['avg_loss']:.2%}
        Average Trade Duration: {metrics['avg_duration']:.1f} days
        ===============================================
        """
        
        return summary