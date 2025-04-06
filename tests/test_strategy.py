import numpy as np
import pandas as pd
from strategy import generate_signals, backtest_strategy

def test_generate_signals():
    predictions = np.array([100, 101, 102, 101, 100])
    signals = generate_signals(predictions, threshold=0.5)
    # Expected: first value 0, then positive differences yield 1 and negative yield -1
    expected = np.array([0, 1, 1, -1, -1])
    np.testing.assert_array_equal(signals, expected)

def test_backtest_strategy():
    data = pd.DataFrame({
        'Price': [100, 102, 101, 103, 105]
    })
    signals = np.array([0, 1, -1, 1, 0])
    result = backtest_strategy(data, signals)
    # Check that cumulative return columns are created
    assert 'Cumulative_Market' in result.columns
    assert 'Cumulative_Strategy' in result.columns
