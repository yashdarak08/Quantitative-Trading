from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, lstm_units=50, dropout_rate=0.2, model_type='LSTM'):
    """
    Build and compile an RNN or LSTM model for time series forecasting.
    
    Parameters:
        input_shape (tuple): Shape of the input data (window_size, features)
        lstm_units (int): Number of hidden units in the RNN/LSTM layers
        dropout_rate (float): Dropout rate for regularization
        model_type (str): 'LSTM' or 'RNN'
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units))
    elif model_type == 'RNN':
        model.add(SimpleRNN(lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(SimpleRNN(lstm_units))
    else:
        raise ValueError("Invalid model_type. Choose 'LSTM' or 'RNN'.")
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
