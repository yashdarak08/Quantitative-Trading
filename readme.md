# Quantitative Trading

This repository contains a quantitative trading project that leverages Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) architectures to forecast financial time series. The goal is to capture temporal dependencies in non-stationary financial indicators and generate trading signals that optimize systematic trading strategies—resulting in improved benchmark returns on indices such as NIFTY50 and S&P500.

## Project Structure

```bash
Quantitative-Trading/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── strategy.py
│   └── main.py
└── tests/
    ├── test_model.py
    └── test_strategy.py
```

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
    - Example configuration:
      ```yaml
      data:
         source: "yahoo"
         ticker: "AAPL"
         start_date: "2010-01-01"
         end_date: "2020-01-01"
    
      model:
         type: "LSTM"
         epochs: 50
         batch_size: 32
    
      strategy:
         threshold: 0.05
      ```

4. **Run the project:** 

    ```bash
    python src/main.py
    ```

