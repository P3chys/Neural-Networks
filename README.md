Stock Prediction Pipeline
This repository contains a machine learning pipeline for predicting future stock prices based on historical market data and financial statements. The pipeline is built using Python and leverages popular libraries like pandas, NumPy, scikit-learn, and TensorFlow.
Features

Data collection: Retrieves historical daily price data and quarterly financial statements for a list of stock tickers using the Yahoo Finance API. Calculates additional technical indicators and financial ratios.
Data cleaning: Handles missing values, removes columns with insufficient data, cleans essential market data columns, and aligns quarterly financial data to a daily frequency. Provides a detailed data cleaning report.
Data preprocessing: Groups and scales features by type (price, volume, returns, technical indicators, financials) to prepare for model training. Creates sequences of historical data to predict future prices. Splits data into train, validation and test sets.
Model architecture: Builds a deep learning model using stacked LSTM or GRU layers, with batch normalization, dropout regularization, and dense output layers. Supports configuring the model for different sequence lengths, prediction horizons, and feature sets.
Training pipeline: Trains the model using the processed data with callbacks for early stopping, model checkpointing, learning rate reduction, and TensorBoard logging. Saves the best model weights and training logs.
Model evaluation: Calculates comprehensive evaluation metrics on the test set including MSE, RMSE, MAE, R-squared, MAPE, and directional accuracy. Creates visualizations of actual vs predicted prices and prediction error distributions. Generates an evaluation report.
Prediction: Makes future price predictions using the trained model on new data. Performs inverse scaling to convert predictions back to the original price scale.

Usage

Install the required dependencies:

pip install -r requirements.txt

Run the main pipeline script:

python main.py

This will collect stock data, preprocess it, train the model, evaluate performance on a test set, and output predictions. Logs and model weights will be saved in the training_logs and models directories.

Customize the pipeline by modifying the parameters in main.py such as:


List of stock tickers to collect data for
Date range for historical data
Model architecture and hyperparameters
Sequence length and prediction horizon
Train/validation/test split ratios


Explore the generated visualizations and reports to analyze the model's predictions and performance. Use TensorBoard to track training progress:

tensorboard --logdir=training_logs/tensorboard

Feel free to experiment with different configurations, add new features, or adapt the pipeline to your specific use case. If you encounter any issues or have suggestions for improvement, please open an issue on the repository.