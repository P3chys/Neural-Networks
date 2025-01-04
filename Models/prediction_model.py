import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from typing import Dict, Tuple, Optional

from config import Config
LAYERS = Config.LAYERS

class TemporalAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.attention_weights = Dense(self.hidden_dim, use_bias=False)
        self.attention_score = Dense(1, use_bias=False)
        super().build(input_shape)
        
    def call(self, inputs):
        # Compute attention weights
        weights = self.attention_weights(inputs)
        weights = tf.nn.tanh(weights)
        weights = self.attention_score(weights)
        weights = tf.nn.softmax(weights, axis=1)
        
        # Apply attention
        return inputs * weights
    
class StockPredictionModel:
    def __init__(self, input_shape, scalers, feature_columns, sequence_length=10):
        self.model = None
        self.input_shape = input_shape
        self.scalers = scalers
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        
    def build_model(self):
        self.model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # First LSTM layer with attention
            LSTM(units=LAYERS[0]['units'], return_sequences=LAYERS[0]['return_sequences']),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=LAYERS[0]['hidden_dim']),
            Dropout(LAYERS[0]['dropout']),
            
            # Second LSTM layer with attention
            LSTM(units=LAYERS[1]['units'], return_sequences=LAYERS[1]['return_sequences']),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=LAYERS[1]['hidden_dim']),
            Dropout(LAYERS[1]['dropout']),
            
            # Third LSTM layer with attention
            LSTM(units=LAYERS[2]['units'], return_sequences=LAYERS[2]['return_sequences']),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=LAYERS[2]['hidden_dim']),
            LSTM(units=32, return_sequences=False),
            Dropout(LAYERS[2]['dropout']),
            
            # Dense layers
            Dense(units=32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(units=16, activation='relu'),
            BatchNormalization(),
            
            Dense(units=1)
        ])
        
        # Compile with custom learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber'  # Huber loss is more robust to outliers
        )
        
        return self.model
    
    def _create_callbacks(self, checkpoint_path: str) -> list:

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        """
        Create training callbacks with adjusted parameters.
        """
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            
            ModelCheckpoint(
                filepath=checkpoint_path+'.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),

            TensorBoard(
                log_dir=f"Data/training_logs/tensorboard/{timestamp}",
                histogram_freq=1
            )
        ]
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=32,
              checkpoint_path: str = 'best_model.h5') -> Dict:
        
        callbacks = self._create_callbacks(checkpoint_path)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model predictions and return metrics
        """
        # Make predictions
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred = self.scalers['Close'].inverse_transform(y_pred_scaled)
        y_true = self.scalers['Close'].inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage accuracy
        accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate prediction trend accuracy
        actual_trends = np.diff(y_true.flatten()) > 0
        predicted_trends = np.diff(y_pred.flatten()) > 0
        trend_accuracy = np.mean(actual_trends == predicted_trends) * 100
        
        # Create evaluation dictionary
        eval_metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2,
            'Price Accuracy (%)': accuracy,
            'Trend Accuracy (%)': trend_accuracy
        }
        
        return eval_metrics, y_pred, y_true
    
    def predict(self, current_data: np.ndarray) -> np.ndarray:
        """
        Predict the next day's closing price
        """
        # Prepare the features
        prepared_data = pd.DataFrame()
        for feature in self.feature_columns:
            if feature in current_data.columns:
                prepared_data[feature] = current_data[feature]
        
        # Scale the features
        scaled_features = {}
        for feature in self.feature_columns:
            scaled_features[feature] = self.scalers[feature].transform(
                prepared_data[feature].values.reshape(-1, 1)
            )
        
        # Combine scaled features
        combined_scaled = np.hstack([scaled_features[feature] for feature in self.feature_columns])
        
        # Make prediction
        prediction = self.model.predict(combined_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1))
        
        # Inverse transform prediction
        return self.scalers['Close'].inverse_transform(prediction)[0][0]
    
    def get_model_summary(self):
        """Print the model architecture summary."""
        return self.model.summary()
    
    def save_model(self, filepath: str):
        """Save the complete model to disk."""
        self.model.save(filepath)
    
    def load_model(self, weights_path: str):
        """Load weight model from file (.h5)."""
        # model = tf.keras.models.load_model(filepath)
        # https://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras
        self.model.load_weights(weights_path) 