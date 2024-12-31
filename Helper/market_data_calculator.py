import pandas as pd
class MarketDataCalculator:
    def __init__(self, market_data):
        self.market_data = market_data

    def calculate_technical_indicators(self):
        self.market_data.index = pd.to_datetime(self.market_data.index).tz_localize(None)
        self.market_data['Market_Cap'] = self.market_data['Close'] * self.market_data['Volume']
        self.market_data['Daily_Return'] = self.market_data['Close'].pct_change()
        self.market_data['Volatility'] = self.market_data['Daily_Return'].rolling(window=30).std()
        self.market_data['MA50'] = self.market_data['Close'].rolling(window=50).mean()
        self.market_data['MA200'] = self.market_data['Close'].rolling(window=200).mean()
        return self.market_data

