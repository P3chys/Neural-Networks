import yfinance as yf
import pandas as pd
import time
from typing import List, Dict, Optional
from Helper.date_validator import DateValidator
from Helper.market_data_calculator import MarketDataCalculator
from Helper.logger import Logger

from config import Config 
STOCK_TICKER = Config.STOCK_TICKER
CHCK1_TICKER = Config.CHCK1_TICKER
CHCK2_TICKER = Config.CHCK2_TICKER

class StockDataCollector:
    def __init__(self, 
                 start_date: str = "2014-12-31", 
                 end_date: str = "2024-12-31", 
                 default_tickers: Optional[List[str]] = ["JNJ","SPY","XLV"]
                 ):
        
        self.logger = Logger('Data/training_logs/training.log')
        # Validate and store the dates
        self.start_date, self.end_date = DateValidator.validate_date_format(start_date, end_date)
        self.daily_stock_data = {}

        if default_tickers is None:
            raise ValueError("Please provide a list of tickers.")
        else:
            self.default_tickers = default_tickers
            self.logger.log_info(f"Stock data collector initialized with tickers: {self.default_tickers}")
            
    

    def collect_daily_data(self) -> Dict[str, pd.DataFrame]:
        jnj = yf.download(STOCK_TICKER, start=self.start_date, end=self.end_date)
        time.sleep(1)
        spy = yf.download(CHCK1_TICKER, start=self.start_date, end=self.end_date)
        time.sleep(1)
        xlv = yf.download(CHCK2_TICKER, start=self.start_date, end=self.end_date)
        
        jnj.index = jnj.index.tz_localize(None)
        spy.index = spy.index.tz_localize(None)
        xlv.index = xlv.index.tz_localize(None)

        common_dates = jnj.index.intersection(spy.index).intersection(xlv.index)

        self.daily_stock_data = jnj.loc[common_dates].copy()
        spy = spy.loc[common_dates]
        xlv = xlv.loc[common_dates]
        
        # Calculate technical indicators
        calc = MarketDataCalculator(jnj, spy, xlv)
        self.daily_stock_data = calc.make_calculations()
        
        self.daily_stock_data = self.daily_stock_data.dropna()
        
        return self.daily_stock_data