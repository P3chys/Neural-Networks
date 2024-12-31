class Config:
    START_DATE = "2010-01-01"
    END_DATE = "2015-12-31"
    MIN_NON_NULL_RATIO = 0.7
    LOG_FILE_PATH = 'training_logs/training.log'
    RAW_DATA_PATH = 'raw_data/'
    SEQUENCE_LENGTH = 100
    PREDICTION_HORIZON = 3
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    STOCK_TICKER = 'JNJ'
    STOCK_TICKERS = ["AAPL", "MSFT", "KO", "CSCO", "AMZN", "NKE", "GOOG"],
    PCA_NUM = 2