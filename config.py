class Config:
    START_DATE = "2010-01-01"
    END_DATE = "2024-12-31"

    LOG_FILE_PATH = 'Data/training_logs/training.log'
    RAW_DATA_PATH = '/Data/raw_data/'
    MODEL_PATH = 'Models/models/best_model'
    GRAPH_PATH = 'Documentation/'

    SEQUENCE_LENGTH = 100
    STOCK_TICKER = 'JNJ',
    CHCK1_TICKER = 'SPY',
    CHCK2_TICKER = 'XLV',

    SELECTED_MODEL = '',
    NEW_MODEL_NAME = 'Model-JNJ'

    # DATA ANALYZER
    ANALYZE = False,
    PCA_NUM = 3
    SAVE_GRAPH = True
    SHOW_GRAPH = False
    SAVE_STATS = True
    PRINT_STATS = True

    # PREDICTION MODEL
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