import numpy as np

class MarketDataCalculator:
    def __init__(self, market_data, spy, xlv):
        self.df = market_data
        self.spy = spy
        self.xlv = xlv

    def make_calculations(self):
        self._calculate_technical_indicators()
        self._calculate_base_features()
        self._calculate_boiling_bands()
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_volatility()
        self._calculate_market_correlation()
        self._calculate_price_range_features()
        return self.df
    
    def _calculate_technical_indicators(self):
        self.df['MA5'] = self.df['Close'].rolling(window=5).mean()
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
        
    def _calculate_base_features(self):
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['LogReturns'] = np.log(self.df['Close']/self.df['Close'].shift(1))
        self.df['RelativeVolume'] = self.df['Volume'] / self.df['Volume'].rolling(window=20).mean()
        
    def _calculate_boiling_bands(self):
        self.df['BB_middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['BB_std'] = self.df['Close'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (2 * self.df['BB_std'])
        self.df['BB_lower'] = self.df['BB_middle'] - (2 * self.df['BB_std'])
        self.df['BB_width'] = (self.df['BB_upper'] - self.df['BB_lower']) / self.df['BB_middle']
        self.df = self.df.drop('BB_std', axis=1)
    
    def _calculate_rsi(self, window_length: int = 14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def _calculate_macd(self):
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['Signal_Line']

    def _calculate_volatility(self):
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        self.df['MomentumRatio'] = self.df['Close'] / self.df['Close'].shift(10)

    def _calculate_market_correlation(self):
        self.df['SPY_Returns'] = self.spy['Close'].pct_change()
        rolling_covariance = self.df['Returns'].rolling(window=20).cov(self.df['SPY_Returns'])
        rolling_variance = self.df['SPY_Returns'].rolling(window=20).var()
        self.df['Market_Beta'] = rolling_covariance / rolling_variance
        self.df['XLV_Returns'] = self.xlv['Close'].pct_change()
        self.df['Sector_RelativeStrength'] = self.df['Returns'] - self.df['XLV_Returns']

    def _calculate_price_range_features(self):
        self.df['Daily_Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        self.df['Gap_Up'] = (self.df['Open'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        self.df['Price_StdDev'] = self.df['Close'].rolling(window=20).std()