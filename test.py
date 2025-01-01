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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

@DeprecationWarning
class PriceConstraint(tf.keras.layers.Layer):
    def __init__(self, max_move_percent=3.0):
        super().__init__()
        self.max_move = max_move_percent/100
        
    def call(self, inputs):
        # Constrain price movements to realistic ranges
        previous_price = inputs[:,-1]
        max_change = previous_price * self.max_move
        return tf.clip_by_value(inputs, previous_price - max_change, 
                              previous_price + max_change)
    
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

class JNJStockPredictor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.model = None
        self.scalers = {}
        self.feature_columns = [
            'Close', 'Volume', 'Returns', 'RelativeVolume',
            'MA5', 'MA20', 'MA50', 'BB_width', 'RSI',
            'MACD', 'MACD_Hist', 'Volatility', 'MomentumRatio',
            'Market_Beta', 'Sector_RelativeStrength', 'Daily_Range',
            'Gap_Up', 'Price_StdDev'
        ]

    def download_and_prepare_data(self, years=10):
        """
        Download and prepare JNJ stock data with advanced features
        """
        # Download JNJ data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        # Download both stocks and ensure timezone consistency
        jnj = yf.download('JNJ', start=start_date, end=end_date)
        spy = yf.download('SPY', start=start_date, end=end_date)
        xlv = yf.download('XLV', start=start_date, end=end_date)
        
        # Convert all indexes to timezone-naive
        jnj.index = jnj.index.tz_localize(None)
        spy.index = spy.index.tz_localize(None)
        xlv.index = xlv.index.tz_localize(None)
        
        # Ensure all dataframes have the same dates
        common_dates = jnj.index.intersection(spy.index).intersection(xlv.index)
        df = jnj.loc[common_dates].copy()
        spy = spy.loc[common_dates]
        xlv = xlv.loc[common_dates]
        
        # Calculate base features
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close']/df['Close'].shift(1))
        df['RelativeVolume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
        df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df = df.drop('BB_std', axis=1)
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Volatility and momentum features
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['MomentumRatio'] = df['Close'] / df['Close'].shift(10)
        
        # Market correlation (using SPY)
        df['SPY_Returns'] = spy['Close'].pct_change()
        rolling_covariance = df['Returns'].rolling(window=20).cov(df['SPY_Returns'])
        rolling_variance = df['SPY_Returns'].rolling(window=20).var()
        df['Market_Beta'] = rolling_covariance / rolling_variance
        
        # Healthcare sector relative strength
        df['XLV_Returns'] = xlv['Close'].pct_change()
        df['Sector_RelativeStrength'] = df['Returns'] - df['XLV_Returns']
        
        # Price range features
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Gap_Up'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Price_StdDev'] = df['Close'].rolling(window=20).std()
        #df['Trend_Strength'] = (df['MA20'] - df['MA50'])
        # Drop any rows with NaN values
        df = df.dropna()
        
        return df
    
    def prepare_sequences(self, df):
        """
        Prepare sequences for training with multiple features
        """
        # Scale each feature independently
        scaled_data = {}
        for feature in self.feature_columns:
            self.scalers[feature] = MinMaxScaler()
            scaled_data[feature] = self.scalers[feature].fit_transform(df[feature].values.reshape(-1, 1))
        
        # Combine all scaled features
        combined_scaled = np.hstack([scaled_data[feature] for feature in self.feature_columns])
        
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(combined_scaled[i:(i + self.sequence_length)])
            y.append(scaled_data['Close'][i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build an enhanced LSTM model with attention mechanism
        """
        self.model = Sequential([
            # Input layer
            Input(shape=input_shape),
            
            # First LSTM layer with attention
            LSTM(units=128, return_sequences=True),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=64),
            Dropout(0.3),
            
            # Second LSTM layer with attention
            LSTM(units=128, return_sequences=True),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=64),
            Dropout(0.3),
            
            # Third LSTM layer with attention
            LSTM(units=64, return_sequences=True),
            LayerNormalization(),
            TemporalAttentionLayer(hidden_dim=32),
            LSTM(units=32, return_sequences=False),
            Dropout(0.3),
            
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
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=32):
        """
        Train the model with advanced callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Stop if no improvement after 5 epochs
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ModelCheckpoint(
                'jnj_best_model.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_next_day(self, current_data):
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
    
    def evaluate_predictions(self, X_test, y_test, df_test):
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

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = JNJStockPredictor()
    
    print("Downloading and preparing data...")
    df = predictor.download_and_prepare_data()
    
    print("Preparing sequences...")
    X, y = predictor.prepare_sequences(df)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    df_test = df.iloc[split_idx + predictor.sequence_length:]
    
    print("Building and training model...")
    predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
    history = predictor.train(X_train, y_train)
    
    print("\nEvaluating model performance...")
    metrics, predictions, actual = predictor.evaluate_predictions(X_test, y_test, df_test)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nMaking prediction for next day...")
    last_data = df.tail(predictor.sequence_length)
    next_day_prediction = predictor.predict_next_day(last_data)
    print(f"Current closing price: ${float(df['Close'].iloc[-1]):.2f}")
    print(f"Predicted next day closing price: ${float(next_day_prediction):.2f}")
    
    # Print last 5 predictions vs actual values
    print("\nLast 5 predictions vs actual values:")
    for i in range(-5, 0):
        print(f"Actual: ${actual[i][0]:.2f} | Predicted: ${predictions[i][0]:.2f} | " 
              f"Difference: ${abs(actual[i][0] - predictions[i][0]):.2f} "
              f"({abs(actual[i][0] - predictions[i][0])/actual[i][0]*100:.2f}%)")