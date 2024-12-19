import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import List, Dict, Optional

class StockDataCollector:
    def __init__(self, start_date: str = "2010-01-01", 
                 end_date: str = "2015-12-31", default_tickers: Optional[List[str]] = None):
        """
        Initialize the stock data collector with date validation.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        # Validate and store the dates
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format. Error: {e}")
            
        self.start_date = start_date
        self.end_date = end_date
        if default_tickers is None:
            self.default_tickers = [
                'AAPL', 'MSFT', 'KO', 'CSCO', 'BX',
                'O', 'AMZN', 'ABBV', 'NKE', 'GOOG'
            ]
        else:
            self.default_tickers = default_tickers

    def _process_financial_data(self, data: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Process financial data safely, handling date conversion and reindexing.
        
        Args:
            data: Financial data DataFrame from yfinance
            daily_index: DatetimeIndex to reindex the data to
            
        Returns:
            Processed DataFrame with financial data aligned to daily dates
        """
        if data.empty:
            return pd.DataFrame(index=daily_index)
            
        # Convert column headers (which are dates) to datetime
        data.columns = pd.to_datetime(data.columns)
        
        # Create a new DataFrame with daily dates
        daily_data = pd.DataFrame(index=daily_index)
        
        # Process each financial metric
        for metric in data.index:
            # Forward fill the quarterly data to daily frequency
            daily_values = data.loc[metric].reindex(daily_index, method='ffill')
            daily_data[f'Financial_{metric}'.replace(' ', '_')] = daily_values
            
        return daily_data

    def collect_daily_data(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect daily stock data with improved financial data handling.
        """
        if tickers is None:
            tickers = self.default_tickers
            
        stock_data = {}
        
        for ticker in tickers:
            print(f"Collecting data for {ticker}...")
            try:
                # Create ticker object
                stock = yf.Ticker(ticker)
                
                # Get daily market data
                market_data = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d"
                )
                
                # Ensure the index is datetime without timezone
                market_data.index = pd.to_datetime(market_data.index).tz_localize(None)
                
                # Calculate technical indicators
                market_data['Market_Cap'] = market_data['Close'] * market_data['Volume']
                market_data['Daily_Return'] = market_data['Close'].pct_change()
                market_data['Volatility'] = market_data['Daily_Return'].rolling(window=30).std()
                market_data['MA50'] = market_data['Close'].rolling(window=50).mean()
                market_data['MA200'] = market_data['Close'].rolling(window=200).mean()
                
                # Get and process quarterly financial data
                try:
                    financials_daily = self._process_financial_data(
                        stock.quarterly_financials, 
                        market_data.index
                    )
                    market_data = pd.concat([market_data, financials_daily], axis=1)
                except Exception as e:
                    print(f"Warning: Could not process financial data for {ticker}: {str(e)}")
                
                # Get and process balance sheet data
                try:
                    balance_sheet_daily = self._process_financial_data(
                        stock.quarterly_balance_sheet,
                        market_data.index
                    )
                    market_data = pd.concat([market_data, balance_sheet_daily], axis=1)
                except Exception as e:
                    print(f"Warning: Could not process balance sheet data for {ticker}: {str(e)}")
                
                # Calculate financial ratios where possible
                try:
                    if ('Financial_Net_Income' in market_data.columns and 
                        'Financial_Total_Assets' in market_data.columns):
                        market_data['ROA'] = (market_data['Financial_Net_Income'] / 
                                            market_data['Financial_Total_Assets'])
                except Exception as e:
                    print(f"Warning: Could not calculate financial ratios for {ticker}: {str(e)}")
                
                # Handle missing values
                market_data = market_data.infer_objects(copy=True)
                
                # Store the processed data
                stock_data[ticker] = market_data
                
                # Add delay to avoid hitting API limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error collecting data for {ticker}: {str(e)}")
                continue
        
        return stock_data
    
    def save_data(self, stock_data: Dict[str, pd.DataFrame], path: str = "stock_data/"):
        """Save the collected data to CSV files with error handling."""
        import os
        os.makedirs(path, exist_ok=True)
        
        for ticker, data in stock_data.items():
            try:
                filename = os.path.join(path, f"{ticker}_daily_data.csv")
                data.to_csv(filename)
                print(f"Successfully saved data for {ticker} to {filename}")
            except Exception as e:
                print(f"Error saving data for {ticker}: {str(e)}")

    def load_data(self, path: str = "stock_data/") -> Dict[str, pd.DataFrame]:
        """Load the saved data with error handling."""
        import os
        
        stock_data = {}
        
        for file in os.listdir(path):
            if file.endswith("_daily_data.csv"):
                ticker = file.replace("_daily_data.csv", "")
                filename = os.path.join(path, file)
                try:
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                    stock_data[ticker] = df
                    print(f"Successfully loaded data for {ticker}")
                except Exception as e:
                    print(f"Error loading data for {ticker}: {str(e)}")
                    
        return stock_data