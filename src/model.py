import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    """
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_dim
        
        # Get last time step output
        out = out[:, -1, :]  # out: batch_size, hidden_dim
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)  # out: batch_size, output_dim
        
        return out

class RNNModel(nn.Module):
    """
    Simple RNN model for time series forecasting
    """
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout_rate=0.2):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: batch_size, seq_length, hidden_dim
        
        # Get last time step output
        out = out[:, -1, :]  # out: batch_size, hidden_dim
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)  # out: batch_size, output_dim
        
        return out

class GRUModel(nn.Module):
    """
    GRU model for time series forecasting
    """
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: batch_size, seq_length, hidden_dim
        
        # Get last time step output
        out = out[:, -1, :]  # out: batch_size, hidden_dim
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)  # out: batch_size, output_dim
        
        return out

def build_model(input_shape, lstm_units=50, dropout_rate=0.2, model_type='LSTM', num_layers=2):
    """
    Build and initialize a PyTorch model for time series forecasting.
    
    Parameters:
        input_shape (tuple): Shape of the input data (window_size, features)
        lstm_units (int): Number of hidden units in the RNN/LSTM/GRU layers
        dropout_rate (float): Dropout rate for regularization
        model_type (str): 'LSTM', 'RNN', or 'GRU'
        num_layers (int): Number of recurrent layers
    
    Returns:
        model: PyTorch model
    """
    # Extract dimensions from input shape
    window_size, features = input_shape
    
    if model_type == 'LSTM':
        model = LSTMModel(
            input_dim=features,
            hidden_dim=lstm_units,
            num_layers=num_layers,
            output_dim=1,
            dropout_rate=dropout_rate
        )
    elif model_type == 'RNN':
        model = RNNModel(
            input_dim=features,
            hidden_dim=lstm_units,
            num_layers=num_layers,
            output_dim=1,
            dropout_rate=dropout_rate
        )
    elif model_type == 'GRU':
        model = GRUModel(
            input_dim=features,
            hidden_dim=lstm_units,
            num_layers=num_layers,
            output_dim=1,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError("Invalid model_type. Choose 'LSTM', 'RNN', or 'GRU'.")
    
    return model

class ModelTrainer:
    """
    A class to handle model training, validation, and prediction
    """
    def __init__(self, model, learning_rate=0.001, device=None):
        self.model = model
        
        # Determine device (GPU/CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize history
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, verbose=1):
        """
        Train the model
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation targets
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            verbose (int): Verbosity level (0=silent, 1=progress bar)
            
        Returns:
            dict: Training history
        """
        # Create datasets and dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            has_validation = True
        else:
            has_validation = False
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            # Batch training
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # Calculate average training loss
            train_loss = train_loss / len(train_dataset)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if has_validation:
                val_loss = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose > 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f}")
        
        return self.history
    
    def _validate(self, val_loader):
        """
        Validate the model
        
        Parameters:
            val_loader: DataLoader for validation data
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
        
        return val_loss / len(val_loader.dataset)
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predictions
        """
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Convert to numpy
        return predictions.cpu().numpy().flatten()
    
    def save_model(self, path):
        """
        Save model to file
        
        Parameters:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from file
        
        Parameters:
            path (str): Path to the saved model
        """
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")


# For compatibility with TensorFlow-style code
class ModelWrapper:
    """
    Wrapper class to provide a similar interface to TensorFlow models
    """
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.history = None
    
    def fit(self, X_train, y_train, validation_data=None, epochs=50, batch_size=32, verbose=1):
        """
        Train the model (TensorFlow-like interface)
        
        Parameters:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            validation_data (tuple, optional): Tuple of (X_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
            
        Returns:
            self: For method chaining
        """
        if validation_data is not None:
            X_val, y_val = validation_data
        else:
            X_val, y_val = None, None
        
        self.history = self.trainer.train(
            X_train, y_train, X_val, y_val, 
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=verbose
        )
        
        return self
    
    def predict(self, X):
        """
        Make predictions (TensorFlow-like interface)
        
        Parameters:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predictions
        """
        return self.trainer.predict(X)
    
    def save(self, path):
        """
        Save model (TensorFlow-like interface)
        
        Parameters:
            path (str): Path to save the model
        """
        self.trainer.save_model(path)
    
    @classmethod
    def load_model(cls, path, device=None):
        """
        Load model (TensorFlow-like interface)
        
        Parameters:
            path (str): Path to the saved model
            device (torch.device, optional): Device to load the model on
            
        Returns:
            ModelWrapper: Loaded model wrapper
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Determine model type and structure from saved state
        state_dict = checkpoint['model_state_dict']
        
        # Check for the presence of different layer types to determine model type
        if any('lstm' in key for key in state_dict.keys()):
            model_type = 'LSTM'
        elif any('gru' in key for key in state_dict.keys()):
            model_type = 'GRU'
        else:
            model_type = 'RNN'
        
        # Extract dimensions from state dict
        for key, value in state_dict.items():
            if 'weight' in key and ('rnn' in key or 'lstm' in key or 'gru' in key):
                input_dim = value.shape[1]
                hidden_dim = value.shape[0]
                break
        
        # Count number of layers
        num_layers = sum(1 for key in state_dict.keys() if 'weight_hh' in key)
        
        # Create model
        if model_type == 'LSTM':
            model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        elif model_type == 'GRU':
            model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        else:
            model = RNNModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        # Create trainer
        trainer = ModelTrainer(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.history = checkpoint['history']
        
        # Create wrapper
        wrapper = cls(model, trainer)
        wrapper.history = trainer.history
        
        return wrapper


# Function to maintain compatibility with the rest of the codebase
def build_model_with_wrapper(input_shape, lstm_units=50, dropout_rate=0.2, model_type='LSTM', num_layers=2, learning_rate=0.001):
    """
    Build and initialize a PyTorch model with a TensorFlow-like wrapper
    
    Parameters:
        input_shape (tuple): Shape of the input data (window_size, features)
        lstm_units (int): Number of hidden units in the RNN/LSTM/GRU layers
        dropout_rate (float): Dropout rate for regularization
        model_type (str): 'LSTM', 'RNN', or 'GRU'
        num_layers (int): Number of recurrent layers
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        ModelWrapper: Wrapped PyTorch model with TensorFlow-like interface
    """
    # Build the PyTorch model
    model = build_model(input_shape, lstm_units, dropout_rate, model_type, num_layers)
    
    # Create trainer
    trainer = ModelTrainer(model, learning_rate=learning_rate)
    
    # Create and return wrapper
    return ModelWrapper(model, trainer)