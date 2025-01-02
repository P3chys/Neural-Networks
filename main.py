# main.py
import os

# Import our modules
from Data.data_collector import StockDataCollector
from Data.data_preprocessor import StockDataPreprocessor
from Models.prediction_model import StockPredictionModel
from Tools.pca_analyzer import PCAAnalyzer
from config import Config
from Helper.directory_setup import DirectorySetup
from Helper.result_printer import ResultPrinter

START_DATE = Config.START_DATE
END_DATE = Config.END_DATE
LOG_FILE_PATH = Config.LOG_FILE_PATH
RAW_DATA_PATH = Config.RAW_DATA_PATH

SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH
STOCK_TICKER = Config.STOCK_TICKER
MODEL_PATH = Config.MODEL_PATH
        
def main():
    ###############################################################
    ###                          SETUP                          ###
    ###############################################################
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    DirectorySetup.setup_directories()

    #has to be after seting up directories
    from Helper.logger import Logger
    from Helper.data_io import DataIO
    
    raw_data_handler = DataIO(RAW_DATA_PATH)
    log = Logger(LOG_FILE_PATH)
    sdc = StockDataCollector(start_date=START_DATE, 
                end_date=END_DATE,
                default_tickers=[STOCK_TICKER])
    
    log.log_info("Starting stock prediction pipeline...")
    
    try:
        ###############################################################
        ###                       COLECT DATA                       ###
        ###############################################################
        log.log_info("Collecting stock data...")
        stock_data = sdc.collect_daily_data()
        raw_data_handler.save_data(stock_data)

        ###############################################################
        ###                     PREPROCESS DATA                     ###
        ###############################################################
        sdp = StockDataPreprocessor(df=stock_data, sequence_length=SEQUENCE_LENGTH)
        log.log_info("Preprocessing cleaned data...")
        X,y = sdp.prepare_data()
        
        ###############################################################
        ###                     SPLIT DATA                          ###
        ###############################################################
        log.log_info("Setting up model training...")
        X_train, X_test, y_train, y_test, df_test = sdp.split_data(X
                                                                   ,y
                                                                   ,stock_data
                                                     ,sequence_length=SEQUENCE_LENGTH)
        
        ###############################################################
        ###                     BUILD & TRAIN MODEL                 ###
        ###############################################################
        spm = StockPredictionModel(input_shape=(X.shape[1], X.shape[2]),scalers=sdp.scalers, feature_columns=sdp.feature_columns, sequence_length=SEQUENCE_LENGTH)
        # 5. Setup Prediction Model
        model = spm.build_model()
        history = spm.train(X_train
                            , y_train
                            , validation_split=0.2
                            , epochs=200
                            , batch_size=32
                            , checkpoint_path=MODEL_PATH)
        
        ###############################################################
        ###                     EVALUATE MODEL                      ###
        ###############################################################
        metrics, predictions, actual = spm.evaluate(X_test, y_test)
        ResultPrinter.print_results(df_test,spm,predictions,actual, metrics)
        
    except Exception as e:
        log.log_error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()