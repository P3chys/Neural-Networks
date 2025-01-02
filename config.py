class Config:
    START_DATE = "2010-01-01"
    END_DATE = "2024-12-31"

    LOG_FILE_PATH = 'Data/training_logs/training.log'
    RAW_DATA_PATH = '/Data/raw_data/'
    MODEL_PATH = 'Models/models/best_model'

    SEQUENCE_LENGTH = 100
    STOCK_TICKER = 'JNJ',
    CHCK1_TICKER = '',
    CHCK2_TICKER = '',

    SELECTED_MODEL = '',
    ANALYZE = False,
    PCA_NUM = 2