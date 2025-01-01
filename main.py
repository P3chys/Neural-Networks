# main.py
import os

# Import our modules
from Data.data_collector import StockDataCollector
from Data.data_preprocessor import StockDataPreprocessor
from Models.prediction_model import StockPredictionModel
from Tools.training_pipeline import StockTrainingPipeline
from Models.model_evaluator import StockModelEvaluator
from Tools.pca_analyzer import PCAAnalyzer
from config import Config
import numpy as np

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
LOG_FILE_PATH = 'Data/training_logs/training.log'
RAW_DATA_PATH = '/Data/raw_data/'
PROCESSED_DATA_PATH = '/Data/preprocessed_data/'
SEQUENCE_LENGTH = 50
STOCK_TICKER = Config.STOCK_TICKER
MODEL_PATH = 'Models/models/best_model'


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'Data/',
        'Models/models/',
        RAW_DATA_PATH,
        'Data/preprocessed_data/',
        'Data/training_logs/',
        'Data/training_logs/tensorboard/'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def pca_analyis(data):
    pca_analyzer = PCAAnalyzer(data)            # Inicializace třídy
    pca_analyzer.scale_data()                   # Škálování dat
    pca_analyzer.perform_pca(Config.PCA_NUM)    # Provedení PCA
    pca_analyzer.visualize_pca()                # Vizualizace výsledků
    pca_analyzer.print_explained_variance()     # Tisk vysvětlené variability
    pca_analyzer.print_components()             # Tisk vlivu atributů

def print_results(df,spm,predictions,actual, metrics):

    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    last_data = df.tail(SEQUENCE_LENGTH)
    next_day_prediction = spm.predict(last_data)
    print(f"Current closing price: ${float(df['Close'].iloc[-1]):.2f}")
    print(f"Predicted next day closing price: ${float(next_day_prediction):.2f}")

    # Print last 5 predictions vs actual values
    print("\nLast 5 predictions vs actual values:")
    for i in range(-5, 0):
        print(f"Actual: ${actual[i][0]:.2f} | Predicted: ${predictions[i][0]:.2f} | " 
              f"Difference: ${abs(actual[i][0] - predictions[i][0]):.2f} "
              f"({abs(actual[i][0] - predictions[i][0])/actual[i][0]*100:.2f}%)")
        
def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    setup_directories()
    from Helper.logger import Logger
    from Helper.data_io import DataIO
    raw_data_handler = DataIO(RAW_DATA_PATH)
    log = Logger(LOG_FILE_PATH)
    sdc = StockDataCollector(start_date=START_DATE, 
                end_date=END_DATE,
                default_tickers=[STOCK_TICKER])
    
    log.log_info("Starting stock prediction pipeline...")
    
    try:
        # 1. Data Collection
        log.log_info("Collecting stock data...")
        stock_data = sdc.collect_daily_data()
        raw_data_handler.save_data(stock_data)

        sdp = StockDataPreprocessor(df=stock_data, sequence_length=SEQUENCE_LENGTH)
        
        log.log_info("Preprocessing cleaned data...")
        X,y = sdp.prepare_data()
        
        # 4. Model Training Setup
        log.log_info("Setting up model training...")
        X_train, X_test, y_train, y_test, df_test = sdp.split_data(X
                                                                   ,y
                                                                   ,stock_data
                                                     ,sequence_length=SEQUENCE_LENGTH)
        
        spm = StockPredictionModel(input_shape=(X.shape[1], X.shape[2]),scalers=sdp.scalers, feature_columns=sdp.feature_columns, sequence_length=SEQUENCE_LENGTH)
        # 5. Setup Prediction Model
        model = spm.build_model()
        history = spm.train(X_train
                            , y_train
                            , validation_split=0.2
                            , epochs=200
                            , batch_size=32
                            , checkpoint_path=MODEL_PATH)
        
        metrics, predictions, actual = spm.evaluate(X_test, y_test)

        print_results(df_test,spm,predictions,actual, metrics)
        
    except Exception as e:
        log.log_error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()