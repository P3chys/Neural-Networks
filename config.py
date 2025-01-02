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
    NEW_MODEL_NAME = 'Model-JNJ'

    ANALYZE = False,
    PCA_NUM = 2

    LAYERS = [
        { #1
            'units': 128,
            'return_sequences': True,
            'hidden_dim': 64,
            'dropout': 0.3
        },
        { #2
            'units': 128,
            'return_sequences': True,
            'hidden_dim': 64,
            'dropout': 0.3
        },
        { #3
            'units': 64,
            'return_sequences': True,
            'hidden_dim': 32,
            'dropout': 0.3
        },
    ]