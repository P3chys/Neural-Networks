# main.py
import os
import logging
from datetime import datetime

# Import our modules
from data_collector import StockDataCollector
from data_cleaner import StockDataCleaner
from data_preprocessor import StockDataPreprocessor
from prediction_model import StockPredictionModel
from training_pipeline import StockTrainingPipeline
from model_evaluator import StockModelEvaluator



START_DATE = "2010-01-01"
END_DATE = "2015-12-31"
MIN_NON_NULL_RATIO = 0.7
LOG_FILE_PATH = 'training_logs/training.log'
RAW_DATA_PATH = 'raw_data/'
SEQUENCE_LENGTH = 100
PREDICTION_HORIZON = 3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
STOCK_TICKER = 'KO'

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'models',
        'raw_data',
        'clean_data',
        'preprocessed_data',
        'training_logs',
        'training_logs/tensorboard'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )

def collect_stock_data():
    try:
        collector = StockDataCollector(
                start_date=START_DATE, 
                end_date=END_DATE,
                default_tickers=[STOCK_TICKER]
                )
        stock_data = collector.collect_daily_data()
        collector.save_data(stock_data=stock_data, path=RAW_DATA_PATH)
        return stock_data
    except Exception as e:
        logging.error(f"Data collection failed: {str(e)}")
        raise

def clean_stock_data(stock_data):
    try:
        cleaner = StockDataCleaner(MIN_NON_NULL_RATIO)
        cleaned_stock_data = cleaner.clean_stock_data(stock_data)
        cleaning_report = cleaner.get_cleaning_report()
        logging.info("\nCleaning Report:")
        logging.info(cleaning_report)
        return cleaned_stock_data
    except Exception as e:
        logging.error(f"Data cleaning failed: {str(e)}")
        raise

def preprocess_stock_data(cleaned_stock_data):
    try:
        preprocessor = StockDataPreprocessor(
            sequence_length=SEQUENCE_LENGTH,
            prediction_horizon=PREDICTION_HORIZON,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO
        )
        prepared_data = preprocessor.prepare_data(cleaned_stock_data)
        return prepared_data, preprocessor
    except Exception as e:
        logging.error(f"Data preprocessing failed: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, preprocessor):
    try:
        evaluator = StockModelEvaluator(model, preprocessor, prediction_horizon=PREDICTION_HORIZON)
        # Calculate metrics
        evaluator.evaluate_predictions(X_test, y_test, STOCK_TICKER)

        # Create visualizations
        evaluator.plot_predictions(X_test, y_test, STOCK_TICKER)
        evaluator.plot_error_distribution(X_test, y_test, STOCK_TICKER)

        # Generate report
        report = evaluator.generate_evaluation_report(STOCK_TICKER)
        print("\nEvaluation Report:")
        print(report.to_string(index=False))
    except Exception as e:
        logging.error(f"Model evaluation failed: {str(e)}")
        raise

def setup_prediction_model(X_train):
    try:
        n_features = X_train.shape[2]
        model = StockPredictionModel(
            sequence_length=SEQUENCE_LENGTH,
            n_features=n_features,
            prediction_horizon=PREDICTION_HORIZON
        )
        return model
    except Exception as e:
        logging.error(f"Model setup failed: {str(e)}")
        raise

def acess_data_splits(prepared_data):
    stock_data = prepared_data[STOCK_TICKER]
    X_train = stock_data['train']['X']
    y_train = stock_data['train']['y']
    X_val = stock_data['val']['X']
    y_val = stock_data['val']['y']
    X_test = stock_data['test']['X']
    y_test = stock_data['test']['y']
    
    # Print data shapes
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}")
    logging.info(f"Test data shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def training_pipeline(model, X_train, y_train, X_val, y_val):
    training_pipeline = StockTrainingPipeline(model)
    history = training_pipeline.train(
        X_train, y_train,
        X_val, y_val,
        epochs=150,
        batch_size=64,
        checkpoint_dir='models'
    )

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Setup project structure
    setup_directories()
    setup_logging()
    
    logging.info("Starting stock prediction pipeline...")
    
    try:
        # 1. Data Collection
        logging.info("Collecting stock data...")
        stock_data = collect_stock_data()
        
        # 2. Data Cleaning
        logging.info("Cleaning collected data...")
        cleaned_stock_data = clean_stock_data(stock_data)
        
        # 3. Data Preprocessing
        logging.info("Preprocessing cleaned data...")
        prepared_data, preprocessor = preprocess_stock_data(cleaned_stock_data)
        
        # 4. Model Training Setup
        logging.info("Setting up model training...")
        X_train, y_train, X_val, y_val, X_test, y_test = acess_data_splits(prepared_data)
        
        
        # 5. Setup Prediction Model
        model = setup_prediction_model(X_train)
        
        # 6. Model Training Pipeline
        logging.info("Starting model training...")
        training_pipeline(model, X_train, y_train, X_val, y_val)
        
        # 7. Model Evaluation
        logging.info("Evaluating model...")
        evaluate_model(model, X_test, y_test, preprocessor)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()