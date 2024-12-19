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
            logging.FileHandler('training_logs/training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Setup project structure
    setup_directories()
    setup_logging()
    
    logging.info("Starting stock prediction pipeline...")
    
    try:
        # 1. Data Collection
        logging.info("Collecting stock data...")
        collector = StockDataCollector(start_date="2010-01-01", end_date="2015-12-31")
        stock_data = collector.collect_daily_data()
        collector.save_data(stock_data=stock_data, path="raw_data/")
        
        # 2. Data Cleaning
        logging.info("Cleaning collected data...")
        cleaner = StockDataCleaner(min_non_null_ratio=0.7)
        cleaned_stock_data = cleaner.clean_stock_data(stock_data)
        cleaning_report = cleaner.get_cleaning_report()
        logging.info("\nCleaning Report:")
        logging.info(cleaning_report)
        
        # 3. Data Preprocessing
        logging.info("Preprocessing cleaned data...")
        preprocessor = StockDataPreprocessor(
            sequence_length=100,      # Use 60 days of history
            prediction_horizon=3,    # Predict 5 days ahead
            train_ratio=0.7,        # 70% for training
            val_ratio=0.15          # 15% for validation (15% for testing)
        )
        
        # Prepare the data
        prepared_data = preprocessor.prepare_data(cleaned_stock_data)
        
        # 4. Model Training (using AAPL as example)
        logging.info("Setting up model training...")
        stock_data = prepared_data['AAPL']
        
        # Access different data splits
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
        
        # Initialize model
        n_features = X_train.shape[2]
        model = StockPredictionModel(
            sequence_length=100,
            n_features=n_features,
            prediction_horizon=3
        )
        
        # Setup and run training pipeline
        logging.info("Starting model training...")
        training_pipeline = StockTrainingPipeline(model)
        history = training_pipeline.train(
            X_train, y_train,
            X_val, y_val,
            epochs=150,
            batch_size=64,
            checkpoint_dir='models'  # Now passing directory instead of specific file path
        )
        
        #Evaluations
        evaluator = StockModelEvaluator(model, preprocessor, prediction_horizon=3)
        # Calculate metrics
        evaluator.evaluate_predictions(X_test, y_test, 'AAPL')

        # Create visualizations
        evaluator.plot_predictions(X_test, y_test, 'AAPL')
        evaluator.plot_error_distribution(X_test, y_test, 'AAPL')

        # Generate report
        report = evaluator.generate_evaluation_report('AAPL')
        print("\nEvaluation Report:")
        print(report.to_string(index=False))
        
        # 6. Make predictions
        logging.info("Making predictions...")
        #test_predictions = model.predict(X_test)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()