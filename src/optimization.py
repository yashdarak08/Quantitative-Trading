import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from strategy import MomentumStrategy, MeanReversionStrategy, MovingAverageCrossoverStrategy, MACDStrategy
from backtest import BacktestEngine

class StrategyOptimizer:
    """
    Optimize strategy parameters using grid search or random search.
    """
    
    def __init__(self, data, strategy_class, param_grid, config=None, n_jobs=1, mode='grid'):
        """
        Initialize the optimizer.
        
        Parameters:
            data (pd.DataFrame): DataFrame with price data
            strategy_class: Strategy class to optimize
            param_grid (dict): Dictionary of parameters and their possible values
            config (dict, optional): Backtest configuration
            n_jobs (int, optional): Number of parallel jobs
            mode (str, optional): 'grid' for grid search, 'random' for random search
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.config = config or {}
        self.n_jobs = n_jobs
        self.mode = mode
        self.results = []
    
    def _evaluate_strategy(self, params):
        """
        Evaluate a strategy with given parameters.
        
        Parameters:
            params (dict): Strategy parameters
            
        Returns:
            dict: Evaluation results
        """
        # Create strategy with parameters
        strategy_config = self.config.copy()
        strategy_config.update(params)
        strategy = self.strategy_class(strategy_config)
        
        # Run backtest
        backtest = BacktestEngine(self.data, strategy, self.config)
        backtest.run()
        
        # Extract relevant metrics
        metrics = backtest.performance_metrics
        
        # Determine the objective metric (can be customized)
        if 'objective' in self.config:
            objective_metric = self.config['objective']
        else:
            # Default to Sharpe ratio
            objective_metric = 'sharpe_ratio_strategy'
        
        # Return results
        result = {
            'params': params,
            'metrics': metrics,
            'objective': metrics[objective_metric]
        }
        
        return result
    
    def _generate_param_combinations(self, n_samples=None):
        """
        Generate parameter combinations based on mode.
        
        Parameters:
            n_samples (int, optional): Number of random samples
            
        Returns:
            list: List of parameter dictionaries
        """
        if self.mode == 'grid':
            # Generate all combinations (grid search)
            keys = self.param_grid.keys()
            values = list(self.param_grid.values())
            combinations = list(product(*values))
            
            return [dict(zip(keys, combo)) for combo in combinations]
        else:
            # Random search
            n_samples = n_samples or 100
            param_list = []
            
            for _ in range(n_samples):
                params = {}
                for key, values in self.param_grid.items():
                    # Handle different parameter types
                    if isinstance(values, list):
                        params[key] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        # Assume (min, max) for range
                        min_val, max_val = values
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[key] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[key] = np.random.uniform(min_val, max_val)
                    elif isinstance(values, tuple) and len(values) == 3:
                        # Assume (min, max, step) for range with step
                        min_val, max_val, step = values
                        values_list = np.arange(min_val, max_val + step, step)
                        params[key] = np.random.choice(values_list)
                
                param_list.append(params)
            
            return param_list
    
    def optimize(self, n_samples=None):
        """
        Perform optimization.
        
        Parameters:
            n_samples (int, optional): Number of random samples for random search
            
        Returns:
            dict: Best parameters and metrics
        """
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(n_samples)
        
        # Evaluate each combination
        if self.n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._evaluate_strategy, params) for params in param_combinations]
                
                for future in as_completed(futures):
                    self.results.append(future.result())
        else:
            # Sequential execution
            for params in param_combinations:
                result = self._evaluate_strategy(params)
                self.results.append(result)
        
        # Find best parameters
        best_result = max(self.results, key=lambda x: x['objective'])
        
        return best_result
    
    def get_top_n_strategies(self, n=10):
        """
        Get the top N performing strategies.
        
        Parameters:
            n (int, optional): Number of top strategies to return
            
        Returns:
            list: List of top N strategy results
        """
        if not self.results:
            return []
        
        # Sort results by objective (descending)
        sorted_results = sorted(self.results, key=lambda x: x['objective'], reverse=True)
        
        # Return top N
        return sorted_results[:n]
    
    def get_param_importance(self):
        """
        Estimate parameter importance based on optimization results.
        
        Returns:
            dict: Parameter importance scores
        """
        if not self.results:
            return {}
        
        # Extract parameters and objectives
        param_names = list(self.results[0]['params'].keys())
        param_values = {name: [] for name in param_names}
        objectives = []
        
        for result in self.results:
            for name in param_names:
                param_values[name].append(result['params'][name])
            objectives.append(result['objective'])
        
        # Convert to DataFrame
        df = pd.DataFrame(param_values)
        df['objective'] = objectives
        
        # Calculate correlation between each parameter and the objective
        importance = {}
        for name in param_names:
            if df[name].dtype in [np.float64, np.int64]:
                importance[name] = abs(df[name].corr(df['objective']))
            else:
                # For categorical parameters, use mean objective for each category
                grouped = df.groupby(name)['objective'].mean()
                # Calculate variance of means (higher variance = more impact)
                importance[name] = grouped.var() / df['objective'].var()
        
        return importance

def optimize_momentum_strategy(data, config=None):
    """
    Optimize momentum strategy parameters.
    
    Parameters:
        data (pd.DataFrame): Price data
        config (dict, optional): Backtest configuration
        
    Returns:
        dict: Optimized strategy parameters and metrics
    """
    # Define parameter grid
    param_grid = {
        'momentum_window': [5, 10, 12, 15, 20, 30],
        'momentum_threshold': [0.0, 0.005, 0.01, 0.02]
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MomentumStrategy,
        param_grid=param_grid,
        config=config
    )
    
    # Run optimization
    best_result = optimizer.optimize()
    
    return best_result

def optimize_mean_reversion_strategy(data, config=None):
    """
    Optimize mean reversion strategy parameters.
    
    Parameters:
        data (pd.DataFrame): Price data
        config (dict, optional): Backtest configuration
        
    Returns:
        dict: Optimized strategy parameters and metrics
    """
    # Define parameter grid
    param_grid = {
        'mean_reversion_window': [10, 15, 20, 30, 40],
        'mean_reversion_threshold': [1.0, 1.5, 2.0, 2.5]
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MeanReversionStrategy,
        param_grid=param_grid,
        config=config
    )
    
    # Run optimization
    best_result = optimizer.optimize()
    
    return best_result

def optimize_moving_average_strategy(data, config=None):
    """
    Optimize moving average crossover strategy parameters.
    
    Parameters:
        data (pd.DataFrame): Price data
        config (dict, optional): Backtest configuration
        
    Returns:
        dict: Optimized strategy parameters and metrics
    """
    # Define parameter grid
    param_grid = {
        'fast_ma_window': [10, 20, 50, 100],
        'slow_ma_window': [50, 100, 200, 300]
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MovingAverageCrossoverStrategy,
        param_grid=param_grid,
        config=config
    )
    
    # Run optimization
    best_result = optimizer.optimize()
    
    return best_result

def optimize_macd_strategy(data, config=None):
    """
    Optimize MACD strategy parameters.
    
    Parameters:
        data (pd.DataFrame): Price data
        config (dict, optional): Backtest configuration
        
    Returns:
        dict: Optimized strategy parameters and metrics
    """
    # Define parameter grid
    param_grid = {
        'macd_fast_period': [8, 10, 12, 15],
        'macd_slow_period': [20, 26, 30],
        'macd_signal_period': [7, 9, 12]
    }
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        data=data,
        strategy_class=MACDStrategy,
        param_grid=param_grid,
        config=config
    )
    
    # Run optimization
    best_result = optimizer.optimize()
    
    return best_result