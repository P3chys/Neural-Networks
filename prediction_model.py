import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from typing import Dict, Tuple, Optional

class StockPredictionModel:
    def __init__(self, 
                 sequence_length: int,
                 n_features: int,
                 prediction_horizon: int,
                 use_gru: bool = False):
        """
        Initialize the stock prediction model with enhanced configuration.
        
        Args:
            sequence_length: Number of timesteps in input sequences
            n_features: Number of features per timestep
            prediction_horizon: Number of future timesteps to predict
            use_gru: Whether to use GRU instead of LSTM (default: False)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_horizon = prediction_horizon
        self.use_gru = use_gru
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> keras.Model:
        """
        Build and return the neural network model with enhanced architecture.
        """
        model = Sequential([
            # First recurrent layer with larger number of units
            LSTM(128, 
                 return_sequences=True,
                 input_shape=(self.sequence_length, self.n_features),
                 kernel_regularizer=L2(0.01)) if not self.use_gru else
            GRU(128,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                kernel_regularizer=L2(0.01)),
                
            BatchNormalization(),
            Dropout(0.3),
            
            # Second recurrent layer
            LSTM(64, return_sequences=False) if not self.use_gru else
            GRU(64, return_sequences=False),
            
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers for prediction
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer
            Dense(self.prediction_horizon, activation='linear')
        ])
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_callbacks(self, checkpoint_path: str) -> list:
        """
        Create training callbacks for monitoring and optimization.
        """
        return [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              checkpoint_path: str = 'best_model.h5') -> Dict:
        """
        Train the model with early stopping and learning rate adjustment.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            checkpoint_path: Path to save best model weights
            
        Returns:
            Training history
        """
        callbacks = self._create_callbacks(checkpoint_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Returns:
            Tuple of (loss, mae) on test data
        """
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features to predict on
            
        Returns:
            Array of predictions
        """
        return self.model.predict(X, verbose=0)
    
    def get_model_summary(self):
        """Print the model architecture summary."""
        return self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the complete model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'StockPredictionModel':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(filepath)
        return model