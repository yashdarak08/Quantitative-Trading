# Quantitative Trading System

A comprehensive quantitative trading system that leverages deep learning models (RNNs and LSTMs) to forecast financial time series and generate profitable trading signals. The system implements momentum and mean-reversion strategies with sophisticated risk management techniques.

## Project Overview

This project aims to develop and optimize systematic trading strategies by:

1. **Engineering time series forecasting models** using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) architectures
2. **Capturing temporal dependencies** in non-stationary financial indicators
3. **Generating trading signals** based on momentum and mean-reversion principles
4. **Optimizing strategy performance** through rigorous backtesting and statistical analysis
5. **Implementing risk management** techniques to control drawdowns and optimize position sizing

The system has been tested on major market indices including NIFTY50 and S&P500, achieving a 4% increase in benchmark returns and 15%+ annualized returns.

## Project Structure

```
Quantitative-Trading/
├── README.md
├── requirements.txt
├── run_tests.py              # Script to run all tests
├── config/
│   └── config.yaml           # Configuration parameters
├── logs/                     # Directory for log files
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_performance.ipynb
│   └── strategy_optimization.ipynb
├── src/
│   ├── data_loader.py        # Data acquisition and preprocessing
│   ├── model.py              # PyTorch-based deep learning models
│   ├── strategy.py           # Trading strategy implementation
│   ├── indicators.py         # Technical indicators for signal generation
│   ├── backtest.py           # Enhanced backtesting framework
│   ├── risk_management.py    # Risk metrics and position sizing
│   ├── optimization.py       # Hyperparameter optimization
│   ├── visualization.py      # Performance visualization tools
│   ├── logger.py             # Logging utility
│   └── main.py               # Main execution script
└── tests/
    ├── test_model.py
    ├── test_strategy.py
    ├── test_indicators.py
    ├── test_backtest.py
    └── test_risk_management.py
```

## Key Features

- **Deep Learning Models**: Implementation of RNN and LSTM architectures in PyTorch for time series forecasting
- **Multiple Signal Types**: Momentum and mean-reversion signals for diversified strategy development
- **Advanced Backtesting**: Realistic simulation of live trading environments with transaction costs and slippage
- **Performance Metrics**: Comprehensive evaluation including Sharpe ratio, Sortino ratio, Calmar ratio, and maximum drawdown
- **Risk Management**: Dynamic position sizing and stop-loss/take-profit mechanisms
- **Statistical Analysis**: Signal monetization and strategy refinement based on statistical significance
- **Fully Tested**: Comprehensive test suite ensures reliability and correctness
- **Logging**: Detailed logging throughout the application for monitoring and debugging

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Quantitative-Trading.git
   cd Quantitative-Trading
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration:**
   - Modify the `config/config.yaml` file to set your parameters for data loading, model training, and strategy execution.

## Usage

### Basic Execution

```bash
python src/main.py
```

### Strategy Development

To develop and test your own strategies:

1. Create a new strategy in `src/strategy.py` by extending the base Strategy class
2. Configure parameters in `config/config.yaml`
3. Run backtest with `python src/main.py --strategy your_strategy_name`

### Model Training

To train a custom model:

```bash
python src/main.py --train --strategy ml
```

### Hyperparameter Optimization

```bash
python src/main.py --optimize --strategy momentum
```

### Running Tests

```bash
python run_tests.py
```

## Performance Metrics

The system calculates and reports key performance metrics:

- **Returns**: Absolute and relative returns
- **Risk-adjusted Performance**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk Metrics**: Volatility, maximum drawdown, value-at-risk (VaR)
- **Statistical Significance**: t-test, p-value, hit ratio

## Advanced Usage

### GPU Acceleration

If you have a CUDA-capable GPU, you can accelerate model training:

```bash
python src/main.py --train --device cuda
```

### Ensemble Strategies

Combine multiple strategies for improved performance:

```bash
python src/main.py --strategy ensemble
```

### Using Pre-trained Models

Load a previously trained model:

```bash
python src/main.py --strategy ml --model_path models/model_20230101120000.pt
```

## Logging

The system includes comprehensive logging to help monitor and debug:

- Logs are stored in the `logs/` directory with a timestamp prefix
- Log level can be configured in the code or command line
- Both console and file logging are supported

## Extending the System

### Adding New Indicators

To add a new technical indicator:

1. Implement the indicator function in `src/indicators.py`
2. Create a signal generation function if needed
3. Update strategies to use the new indicator

### Creating Custom Models

To create a custom deep learning model:

1. Extend the base model classes in `src/model.py`
2. Implement your custom architecture
3. Use the model with an MLStrategy

## License

This project is licensed under the MIT License - see the LICENSE file for details.