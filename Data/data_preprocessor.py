import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, List, Tuple, Optional

class StockDataPreprocessor:
    def __init__(self,df, sequence_length: int = 10):
        self.feature_columns = [
            'Close', 'Volume', 'Returns', 'RelativeVolume',
            'MA5', 'MA20', 'MA50', 'BB_width', 'RSI',
            'MACD', 'MACD_Hist', 'Volatility', 'MomentumRatio',
            'Market_Beta', 'Sector_RelativeStrength', 'Daily_Range',
            'Gap_Up', 'Price_StdDev'
        ]
        self.scalers = {}
        self.df = df
        self.sequence_length = sequence_length
        
    def prepare_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        combined_scaled, scaled = self._scale_features()
        X, y = self._create_sequences(combined_scaled,scaled)
        return X, y
    
    def _scale_features(self):
        scaled_data = {}
        for feature in self.feature_columns:
            self.scalers[feature] = MinMaxScaler()
            scaled_data[feature] = self.scalers[feature].fit_transform(self.df[feature].values.reshape(-1, 1))
        
        combined_scaled = np.hstack([scaled_data[feature] for feature in self.feature_columns])
        return combined_scaled, scaled_data
    
    def _create_sequences(self, combined_scaled,scaled_data) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(self.df) - self.sequence_length):
            X.append(combined_scaled[i:(i + self.sequence_length)])
            y.append(scaled_data['Close'][i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def split_data(self,X,y,df,sequence_length) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        df_test = df.iloc[split_idx + sequence_length:]
        
        return X_train, X_test, y_train, y_test, df_test
    
    @DeprecationWarning
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