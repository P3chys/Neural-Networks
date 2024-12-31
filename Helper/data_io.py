from typing import Dict
import pandas as pd
import os

class DataIO:
    def __init__(self, path: str = "stock_data/"):
        self.path = path

    def save_data(self, data: Dict[str, pd.DataFrame]):
        """Save the collected data to CSV files with error handling."""
        
        os.makedirs(self.path, exist_ok=True)
        for ticker, data in data.items():
            try:
                filename = os.path.join(self.path, f"{ticker}_daily_data.csv")
                data.to_csv(filename)
                print(f"Successfully saved data for {ticker} to {filename}")
            except Exception as e:
                print(f"Error saving data for {ticker}: {str(e)}")

    def load_data(self) -> Dict[str, pd.DataFrame]:

        """Load the saved data with error handling."""
        stock_data = {}
        
        for file in os.listdir(self.path):
            if file.endswith("_daily_data.csv"):
                ticker = file.replace("_daily_data.csv", "")
                filename = os.path.join(self.path, file)
                try:
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                    stock_data[ticker] = df
                    print(f"Successfully loaded data for {ticker}")
                except Exception as e:
                    print(f"Error loading data for {ticker}: {str(e)}")
                    
        return stock_data