import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional

class StockDataPreprocessor:
    def __init__(self, 
                 sequence_length: int = 60,  # Number of days to look back
                 prediction_horizon: int = 5, # Number of days to predict ahead
                 train_ratio: float = 0.7,   # Proportion of data for training
                 val_ratio: float = 0.15):   # Proportion of data for validation
        """
        Initialize the preprocessor for preparing stock data for neural network training.
        
        Args:
            sequence_length: Number of past days to use for prediction
            prediction_horizon: Number of future days to predict
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
                      (remaining data will be used for testing)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scalers = {}  # Store scalers for each feature group
        
    def prepare_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare the cleaned stock data for neural network training.
        
        Args:
            stock_data: Dictionary of cleaned stock DataFrames
            
        Returns:
            Dictionary containing prepared datasets for each stock with train/val/test splits
        """
        prepared_data = {}
        
        for ticker, df in stock_data.items():
            print(f"\nPreparing data for {ticker}...")
            
            # Group features and create feature sets
            feature_groups = self._group_features(df)
            
            # Scale features by group
            scaled_data = self._scale_features(feature_groups, ticker)
            
            # Create sequences and targets
            X, y = self._create_sequences(scaled_data)
            
            # Split data into train/validation/test sets
            train_data, val_data, test_data = self._split_data(X, y)
            
            # Store prepared data
            prepared_data[ticker] = {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'feature_names': list(df.columns)
            }
            
        return prepared_data
    
    def _group_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group features by type for appropriate scaling.
        """
        feature_groups = {
            'price': df[['Open', 'High', 'Low', 'Close']],
            'volume': df[['Volume']],
            'returns': df[['Daily_Return']],
            'technical': df[['Volatility', 'MA50', 'MA200']].select_dtypes(include=[np.number]),
        }
        

            
        return feature_groups
    
    def _scale_features(self, feature_groups: Dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame:
        """
        Scale features appropriately by group.
        """
        scaled_data = pd.DataFrame()
        self.scalers[ticker] = {}
        
        for group_name, group_data in feature_groups.items():
            if group_data.empty:
                continue
                
            scaler = StandardScaler()
            # Reshape to 2D if necessary
            if len(group_data.shape) == 1:
                group_data = group_data.values.reshape(-1, 1)
                
            scaled_values = scaler.fit_transform(group_data)
            self.scalers[ticker][group_name] = scaler
            
            # Create DataFrame with scaled values
            scaled_df = pd.DataFrame(
                scaled_values,
                columns=group_data.columns if hasattr(group_data, 'columns') else [group_name],
                index=group_data.index if hasattr(group_data, 'index') else None
            )
            
            scaled_data = pd.concat([scaled_data, scaled_df], axis=1)
        
        return scaled_data
    
    def _create_sequences(self, 
                         data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of data for training.
        
        Returns:
            X: Array of shape (n_samples, sequence_length, n_features)
            y: Array of shape (n_samples, prediction_horizon)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Get sequence of past data
            sequence = data.iloc[i:(i + self.sequence_length)].values
            
            # Get target future values (using Close price as target)
            target = data.iloc[(i + self.sequence_length):(i + self.sequence_length + self.prediction_horizon)]['Close'].values
            
            X.append(sequence)
            y.append(target)
            
        return np.array(X), np.array(y)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], 
                                                                Dict[str, np.ndarray], 
                                                                Dict[str, np.ndarray]]:
        """
        Split data into train, validation, and test sets while preserving temporal order.
        """
        # Calculate split indices
        n_samples = len(X)
        train_idx = int(n_samples * self.train_ratio)
        val_idx = int(n_samples * (self.train_ratio + self.val_ratio))
        
        # Split the data
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        # Create return dictionaries
        train_data = {'X': X_train, 'y': y_train}
        val_data = {'X': X_val, 'y': y_val}
        test_data = {'X': X_test, 'y': y_test}
        
        return train_data, val_data, test_data
    
    def inverse_transform_predictions(self, predictions: np.ndarray, ticker: str) -> np.ndarray:
        """
        Convert scaled predictions back to original price scale.
        """
        if 'price' not in self.scalers[ticker]:
            raise KeyError(f"No price scaler found for {ticker}")
            
        # Create a dummy array with the same shape as what was used for fitting
        dummy = np.zeros((predictions.shape[0], 4))  # 4 columns for OHLC
        # Put the predictions in the Close price column (assumed to be column 3)
        dummy[:, 3] = predictions.reshape(-1)
        
        # Inverse transform
        original_scale = self.scalers[ticker]['price'].inverse_transform(dummy)
        
        # Return only the Close price column
        return original_scale[:, 3].reshape(predictions.shape)