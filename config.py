class Config:
    START_DATE = "2010-01-01"
    END_DATE = "2024-12-31"

    LOG_FILE_PATH = 'Data/training_logs/training.log'
    RAW_DATA_PATH = '/Data/raw_data/'
    MODEL_PATH = 'Models/models/model_JNJ' # save model
    GRAPH_PATH = 'Documentation/'          # save graph
    SELECTED_MODEL = 'Models/models/model_JNJ.weights.h5'

    LEARN_MODEL = False      # True = training model
    SEQUENCE_LENGTH = 100
    STOCK_TICKER = 'JNJ'    # JNJ, PFE, LLY, MRK
    CHCK1_TICKER = 'SPY'
    CHCK2_TICKER = 'XLV'

    # DATA ANALYZER
    ANALYZE = False,
    PCA_NUM = 3
    SAVE_GRAPH = False
    SHOW_GRAPH = False
    SAVE_STATS = False
    PRINT_STATS = False

    # PREDICTION MODEL
    EPOCHS = 200
    BATCH_SIZE = 32
    VALID_SPLIT = 0.2
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