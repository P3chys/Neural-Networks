
# Neural Networks

This repository contains a machine learning pipeline for predicting future stock prices based on historical market data and financial statements. The pipeline is built using Python and leverages popular libraries like pandas, NumPy, scikit-learn, and TensorFlow. Features




## Usage
Install the required dependencies
```bash
pip install -r requirements.txt
```
Run the main pipeline script:
```bash
python main.py
```

This will collect stock data, preprocess it, train the model, evaluate performance on a test set, and output predictions. Logs and model weights will be saved in the training_logs and models directories.
## Features

- Data collection: Retrieves historical daily price data and quarterly financial statements for a list of stock tickers using the Yahoo Finance API. Calculates additional technical indicators and financial ratios.
- Data cleaning: Handles missing values, removes columns with insufficient data, cleans essential market data columns, and aligns quarterly financial data to a daily frequency. Provides a detailed data cleaning report.
- Data preprocessing: Groups and scales features by type (price, volume, returns, technical indicators, financials) to prepare for model training. Creates sequences of historical data to predict future prices. Splits data into train, validation and test sets.
- Model architecture: Builds a deep learning model using stacked LSTM or GRU layers, with batch normalization, dropout regularization, and dense output layers. Supports configuring the model for different sequence lengths, prediction horizons, and feature sets.
- Training pipeline: Trains the model using the processed data with callbacks for early stopping, model checkpointing, learning rate reduction, and TensorBoard logging. Saves the best model weights and training logs.
- Model evaluation: Calculates comprehensive evaluation metrics on the test set including MSE, RMSE, MAE, R-squared, MAPE, and directional accuracy. Creates visualizations of actual vs predicted prices and prediction error distributions. Generates an evaluation report.
- Prediction: Makes future price predictions using the trained model on new data. Performs inverse scaling to convert predictions back to the original price scale.


## Tracking

Explore the generated visualizations and reports to analyze the model's predictions and performance. Use TensorBoard to track training progress:
```bash
tensorboard --logdir=training_logs/tensorboard
```

