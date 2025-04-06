import numpy as np
from model import build_model

def test_build_model_lstm():
    input_shape = (60, 1)
    model = build_model(input_shape=input_shape, lstm_units=10, dropout_rate=0.1, model_type='LSTM')
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == 1

def test_build_model_rnn():
    input_shape = (60, 1)
    model = build_model(input_shape=input_shape, lstm_units=10, dropout_rate=0.1, model_type='RNN')
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == 1
