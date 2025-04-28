import unittest
import torch
import numpy as np
from src.model import (
    build_model, TimeSeriesDataset, LSTMModel, 
    RNNModel, GRUModel, ModelTrainer, ModelWrapper
)

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic data for testing
        self.window_size = 20
        self.n_features = 1
        self.batch_size = 16
        self.n_samples = 100
        
        # Create random input data
        self.X = np.random.randn(self.n_samples, self.window_size, self.n_features)
        self.y = np.random.randn(self.n_samples)
        
        # Define model parameters
        self.input_shape = (self.window_size, self.n_features)
        self.lstm_units = 32
        self.dropout_rate = 0.2
        self.num_layers = 2
    
    def test_time_series_dataset(self):
        """Test TimeSeriesDataset class"""
        # Create dataset
        dataset = TimeSeriesDataset(self.X, self.y)
        
        # Check length
        self.assertEqual(len(dataset), self.n_samples)
        
        # Check item retrieval
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (self.window_size, self.n_features))
        self.assertEqual(y.shape, ())
        
        # Check all items are retrievable
        for i in range(len(dataset)):
            x, y = dataset[i]
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
    
    def test_lstm_model(self):
        """Test LSTM model architecture"""
        # Create LSTM model
        model = LSTMModel(
            input_dim=self.n_features,
            hidden_dim=self.lstm_units,
            num_layers=self.num_layers,
            output_dim=1,
            dropout_rate=self.dropout_rate
        )
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Forward pass with batch
        x = torch.randn(self.batch_size, self.window_size, self.n_features)
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_rnn_model(self):
        """Test RNN model architecture"""
        # Create RNN model
        model = RNNModel(
            input_dim=self.n_features,
            hidden_dim=self.lstm_units,
            num_layers=self.num_layers,
            output_dim=1,
            dropout_rate=self.dropout_rate
        )
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Forward pass with batch
        x = torch.randn(self.batch_size, self.window_size, self.n_features)
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_gru_model(self):
        """Test GRU model architecture"""
        # Create GRU model
        model = GRUModel(
            input_dim=self.n_features,
            hidden_dim=self.lstm_units,
            num_layers=self.num_layers,
            output_dim=1,
            dropout_rate=self.dropout_rate
        )
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Forward pass with batch
        x = torch.randn(self.batch_size, self.window_size, self.n_features)
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_build_model(self):
        """Test model builder function"""
        # Build LSTM model
        lstm_model = build_model(
            input_shape=self.input_shape,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type='LSTM',
            num_layers=self.num_layers
        )
        self.assertIsInstance(lstm_model, LSTMModel)
        
        # Build RNN model
        rnn_model = build_model(
            input_shape=self.input_shape,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type='RNN',
            num_layers=self.num_layers
        )
        self.assertIsInstance(rnn_model, RNNModel)
        
        # Build GRU model
        gru_model = build_model(
            input_shape=self.input_shape,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type='GRU',
            num_layers=self.num_layers
        )
        self.assertIsInstance(gru_model, GRUModel)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            invalid_model = build_model(
                input_shape=self.input_shape,
                lstm_units=self.lstm_units,
                dropout_rate=self.dropout_rate,
                model_type='INVALID',
                num_layers=self.num_layers
            )
    
    def test_model_trainer(self):
        """Test model trainer class"""
        # Create model and trainer
        model = LSTMModel(
            input_dim=self.n_features,
            hidden_dim=self.lstm_units,
            num_layers=self.num_layers,
            output_dim=1,
            dropout_rate=self.dropout_rate
        )
        trainer = ModelTrainer(model, learning_rate=0.001)
        
        # Check initialization
        self.assertEqual(trainer.model, model)
        
        # Convert data to tensors for prediction
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        
        # Test prediction
        with torch.no_grad():
            predictions = trainer.predict(self.X)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (self.n_samples,))
    
    def test_model_wrapper(self):
        """Test model wrapper class"""
        # Create model and trainer
        model = LSTMModel(
            input_dim=self.n_features,
            hidden_dim=self.lstm_units,
            num_layers=self.num_layers,
            output_dim=1,
            dropout_rate=self.dropout_rate
        )
        trainer = ModelTrainer(model, learning_rate=0.001)
        
        # Create wrapper
        wrapper = ModelWrapper(model, trainer)
        
        # Test prediction
        predictions = wrapper.predict(self.X)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (self.n_samples,))
    
    def test_model_wrapper_compatibility(self):
        """Test ModelWrapper compatibility with TensorFlow-style interfaces"""
        # Create model via wrapper builder
        wrapper = build_model_with_wrapper(
            input_shape=self.input_shape,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            model_type='LSTM',
            num_layers=self.num_layers
        )
        
        # Check wrapper type
        self.assertIsInstance(wrapper, ModelWrapper)
        
        # Try calling TensorFlow-style methods
        self.assertIsNotNone(wrapper.predict)
        self.assertIsNotNone(wrapper.fit)
        self.assertIsNotNone(wrapper.save)
        
        # Test that predict works
        predictions = wrapper.predict(self.X)
        self.assertEqual(predictions.shape, (self.n_samples,))

# Import the builder function we're testing at the bottom
# This is needed because it was referenced in the test but not imported at the top
from src.model import build_model_with_wrapper

if __name__ == "__main__":
    unittest.main()