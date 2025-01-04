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
from Tools.data_analyzer import DataAnalyzer

START_DATE = Config.START_DATE
END_DATE = Config.END_DATE
LOG_FILE_PATH = Config.LOG_FILE_PATH
RAW_DATA_PATH = Config.RAW_DATA_PATH

SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH
STOCK_TICKER = Config.STOCK_TICKER
MODEL_PATH = Config.MODEL_PATH

LEARN_MODEL = Config.LEARN_MODEL
EPOCHS = Config.EPOCHS
BATCH_SIZE = Config.BATCH_SIZE
VALID_SPLIT = Config.VALID_SPLIT

ANALYZE = Config.ANALYZE
SELECTED_MODEL= Config.SELECTED_MODEL # pokud == '' vytvoří se nový
        
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
        ###                      ANALYZE DATA                       ###
        ###############################################################
        if ANALYZE:
            analyzer = DataAnalyzer(data=stock_data)
            analyzer.analyze()

        ###############################################################
        ###                     PREPROCESS DATA                     ###
        ###############################################################
        sdp = StockDataPreprocessor(df=stock_data, sequence_length=SEQUENCE_LENGTH)
        log.log_info("Preprocessing cleaned data...")
        X,y = sdp.prepare_data()
        
        ###############################################################
        ###                     SPLIT DATA                          ###
        ###############################################################
        log.log_info("Setting up model data...")
        X_train, X_test, y_train, y_test, df_test = sdp.split_data(X
                                                                   ,y
                                                                   ,stock_data
                                                     ,sequence_length=SEQUENCE_LENGTH)
        
        ###############################################################
        ###                     BUILD & TRAIN MODEL                 ###
        ###############################################################
        
        # 5. Setup Prediction Model
        log.log_info("Setting up model building...")
        spm = StockPredictionModel(input_shape=(X.shape[1], X.shape[2]),
                                    scalers=sdp.scalers,
                                    feature_columns=sdp.feature_columns,
                                    sequence_length=SEQUENCE_LENGTH)
        model = spm.build_model()

        if SELECTED_MODEL != '':
            log.log_info(f"Loading existing model from: {SELECTED_MODEL}")
            spm.load_model(weights_path=SELECTED_MODEL)
        
        if LEARN_MODEL:
            log.log_info("Setting up model training...")
            history = spm.train(X_train,y_train, 
                            validation_split=VALID_SPLIT, 
                            epochs=EPOCHS, 
                            batch_size=BATCH_SIZE, 
                            checkpoint_path=MODEL_PATH)
        
        ###############################################################
        ###                     EVALUATE MODEL                      ###
        ###############################################################
        
        log.log_info("Evaluating model...")
        metrics, predictions, actual = spm.evaluate(X_test, y_test)
        ResultPrinter.print_results(df_test,spm,predictions,actual, metrics)
        
    except Exception as e:
        log.log_error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()