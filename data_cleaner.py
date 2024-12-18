import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class StockDataCleaner:
    def __init__(self, min_non_null_ratio: float = 0.7):
        """
        Initialize the data cleaner with configuration parameters.
        
        Args:
            min_non_null_ratio: Minimum ratio of non-null values required in a column
                               to keep it in the dataset (default: 0.7 or 70%)
        """
        self.min_non_null_ratio = min_non_null_ratio
        # Store statistics about cleaning process
        self.cleaning_stats = {}
        
    def clean_stock_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean the stock data dictionary containing DataFrames for multiple stocks.
        
        Args:
            stock_data: Dictionary with stock tickers as keys and DataFrames as values
            
        Returns:
            Dictionary with cleaned DataFrames
        """
        cleaned_data = {}
        self.cleaning_stats = {}
        
        for ticker, df in stock_data.items():
            print(f"\nCleaning data for {ticker}...")
            
            # Initialize statistics for this ticker
            self.cleaning_stats[ticker] = {
                'original_columns': len(df.columns),
                'original_rows': len(df),
                'removed_columns': [],
                'null_percentages': {}
            }
            
            # Clean the individual DataFrame
            cleaned_df = self._clean_single_stock(df, ticker)
            
            # Only include stocks with sufficient data
            if not cleaned_df.empty:
                cleaned_data[ticker] = cleaned_df
                # Update final statistics
                self.cleaning_stats[ticker].update({
                    'final_columns': len(cleaned_df.columns),
                    'final_rows': len(cleaned_df)
                })
            else:
                print(f"Warning: Removed {ticker} due to insufficient data")
                
        return cleaned_data
    
    def _clean_single_stock(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Clean a single stock's DataFrame by handling missing values and removing problematic columns.
        
        Args:
            df: DataFrame for a single stock
            ticker: Stock ticker symbol for logging purposes
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original data
        df = df.copy()
        
        # Step 1: Remove columns with too many null values
        null_percentages = df.isnull().mean().round(4)
        self.cleaning_stats[ticker]['null_percentages'] = null_percentages.to_dict()
        
        columns_to_keep = null_percentages[null_percentages <= (1 - self.min_non_null_ratio)].index
        columns_removed = set(df.columns) - set(columns_to_keep)
        
        if columns_removed:
            print(f"Removing columns with > {(1-self.min_non_null_ratio)*100}% null values: {list(columns_removed)}")
            self.cleaning_stats[ticker]['removed_columns'] = list(columns_removed)
            df = df[columns_to_keep]
        
        # Step 2: Handle essential market data columns that should never be null
        essential_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in essential_columns:
            if col in df.columns:
                # Forward fill first, then backward fill for any remaining nulls
                df[col] = df[col].ffill().bfill()
        
        # Step 3: Handle derived technical indicators
        technical_indicators = ['Daily_Return', 'Volatility', 'MA50', 'MA200']
        for col in technical_indicators:
            if col in df.columns:
                # For technical indicators, we'll use forward fill as they're typically rolling calculations
                df[col] = df[col].ffill()
        
        # Step 4: Handle financial metrics
        financial_columns = [col for col in df.columns if col.startswith('Financial_')]
        for col in financial_columns:
            # Financial data is reported quarterly, so we'll forward fill
            df[col] = df[col].ffill()
        
        # Step 5: Clean up any remaining nulls in other columns
        # First forward fill, then backward fill for any remaining nulls
        df = df.ffill().bfill()
        
        # Step 6: Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
        return df
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """
        Generate a detailed report of the cleaning process.
        
        Returns:
            DataFrame containing cleaning statistics for each stock
        """
        report_data = []
        
        for ticker, stats in self.cleaning_stats.items():
            report_row = {
                'Ticker': ticker,
                'Original Columns': stats['original_columns'],
                'Final Columns': stats.get('final_columns', 0),
                'Original Rows': stats['original_rows'],
                'Final Rows': stats.get('final_rows', 0),
                'Removed Columns Count': len(stats['removed_columns']),
                'Removed Columns': ', '.join(stats['removed_columns']) if stats['removed_columns'] else 'None'
            }
            report_data.append(report_row)
            
        return pd.DataFrame(report_data)
    
    def plot_null_percentages(self, ticker: str):
        """
        Create a visualization of null percentages for a given stock.
        
        Args:
            ticker: Stock ticker to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            null_percentages = pd.Series(self.cleaning_stats[ticker]['null_percentages'])
            
            plt.figure(figsize=(12, 6))
            null_percentages.sort_values(ascending=True).plot(kind='bar')
            plt.title(f'Null Percentages by Column for {ticker}')
            plt.xlabel('Columns')
            plt.ylabel('Null Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib is required for plotting. Please install it first.")