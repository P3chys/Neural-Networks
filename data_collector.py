import yfinance as yf
import pandas as pd
import time
from typing import List, Dict, Optional
from Helper.data_io import DataIO
from Helper.date_validator import DateValidator
from Helper.market_data_calculator import MarketDataCalculator

class StockDataCollector:
    def __init__(self, start_date: str = "2010-01-01", 
                 end_date: str = "2015-12-31", 
                 default_tickers: Optional[List[str]] = None
                 ):
        """
        Initialize the stock data collector with date validation.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            default_tickers: List of default tickers to collect data for
        """
        # Validate and store the dates
        self.start_date, self.end_date = DateValidator.validate_date_format(start_date, end_date)
        self.daily_stock_data = {}

        if default_tickers is None:
            raise ValueError("Please provide a list of tickers.")
        else:
            self.default_tickers = default_tickers
    
    def __get_data(self, ticker: str) -> pd.DataFrame:
        """
        Get daily stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame: Daily stock data
        """
        stock = yf.Ticker(ticker)
        data = stock.history(
            start=self.start_date,
            end=self.end_date,
            interval="1d"
        )
        return data


    def collect_daily_data(self) -> Dict[str, pd.DataFrame]:
        for ticker in self.default_tickers:
            print(f"Collecting data for {ticker}...")
            try:
                market_data = self.__get_data(ticker)
                
                # Calculate technical indicators
                calc = MarketDataCalculator(market_data)
                market_data = calc.calculate_technical_indicators()
                
                # Handle missing values
                market_data = market_data.infer_objects(copy=True)
                
                # Store the processed data
                self.daily_stock_data[ticker] = market_data
                
                # Add delay to avoid hitting API limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error collecting data for {ticker}: {str(e)}")
                continue
        
        return self.daily_stock_data