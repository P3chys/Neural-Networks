import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class StockModelEvaluator:
    def __init__(self, model, preprocessor, prediction_horizon: int = 5):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained stock prediction model
            preprocessor: Data preprocessor instance used for scaling
            prediction_horizon: Number of days ahead being predicted
        """
        self.model = model
        self.preprocessor = preprocessor
        self.prediction_horizon = prediction_horizon
        self.evaluation_metrics = {}
        
    def evaluate_predictions(self, 
                           X_test: np.ndarray, 
                           y_test: np.ndarray, 
                           ticker: str) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for model predictions.
        
        Args:
            X_test: Test features
            y_test: Actual test values
            ticker: Stock ticker symbol for inverse scaling
        
        Returns:
            Dictionary containing various evaluation metrics
        """
        # Get model predictions
        y_pred = self.model.predict(X_test)
        
        # Convert scaled values back to original price scale
        y_test_orig = self.preprocessor.inverse_transform_predictions(y_test, ticker)
        y_pred_orig = self.preprocessor.inverse_transform_predictions(y_pred, ticker)
        
        # Calculate various error metrics
        metrics = {
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'r2': r2_score(y_test_orig, y_pred_orig),
            'mape': np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
        }
        
        # Calculate directional accuracy
        direction_test = np.diff(y_test_orig, axis=1)
        direction_pred = np.diff(y_pred_orig, axis=1)
        correct_direction = np.sum(np.sign(direction_test) == np.sign(direction_pred))
        total_directions = direction_test.size
        metrics['directional_accuracy'] = (correct_direction / total_directions) * 100
        
        self.evaluation_metrics[ticker] = metrics
        return metrics
    
    def plot_predictions(self, 
                        X_test: np.ndarray, 
                        y_test: np.ndarray, 
                        ticker: str,
                        samples: int = 100):
        """
        Create visualization of predicted vs actual values.
        
        Args:
            X_test: Test features
            y_test: Actual test values
            ticker: Stock ticker symbol
            samples: Number of samples to plot
        """
        y_pred = self.model.predict(X_test)
        
        # Convert to original scale
        y_test_orig = self.preprocessor.inverse_transform_predictions(y_test, ticker)
        y_pred_orig = self.preprocessor.inverse_transform_predictions(y_pred, ticker)
        
        # Create time index for plotting
        time_index = np.arange(samples)
        
        # Plot predictions vs actual values
        plt.figure(figsize=(15, 8))
        
        # Plot each day in the prediction horizon
        for day in range(self.prediction_horizon):
            plt.subplot(2, 3, day + 1)
            plt.plot(time_index, y_test_orig[:samples, day], 
                    label='Actual', color='blue', alpha=0.7)
            plt.plot(time_index, y_pred_orig[:samples, day], 
                    label='Predicted', color='red', alpha=0.7)
            plt.title(f'Day {day + 1} Predictions')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_error_distribution(self, 
                              X_test: np.ndarray, 
                              y_test: np.ndarray, 
                              ticker: str):
        """
        Plot the distribution of prediction errors.
        """
        y_pred = self.model.predict(X_test)
        
        # Convert to original scale
        y_test_orig = self.preprocessor.inverse_transform_predictions(y_test, ticker)
        y_pred_orig = self.preprocessor.inverse_transform_predictions(y_pred, ticker)
        
        # Calculate errors
        errors = y_pred_orig - y_test_orig
        
        plt.figure(figsize=(15, 5))
        
        # Error distribution
        plt.subplot(1, 2, 1)
        sns.histplot(errors.flatten(), kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Count')
        
        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(errors.flatten(), dist="norm", plot=plt)
        plt.title('Q-Q Plot of Prediction Errors')
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, ticker: str) -> pd.DataFrame:
        """
        Generate a detailed evaluation report.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame containing evaluation metrics
        """
        if ticker not in self.evaluation_metrics:
            raise KeyError(f"No evaluation metrics found for {ticker}")
            
        metrics = self.evaluation_metrics[ticker]
        
        # Create report DataFrame
        report = pd.DataFrame({
            'Metric': [
                'Mean Squared Error (MSE)',
                'Root Mean Squared Error (RMSE)',
                'Mean Absolute Error (MAE)',
                'R-squared (R²)',
                'Mean Absolute Percentage Error (MAPE)',
                'Directional Accuracy'
            ],
            'Value': [
                f"{metrics['mse']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['r2']:.4f}",
                f"{metrics['mape']:.2f}%",
                f"{metrics['directional_accuracy']:.2f}%"
            ]
        })
        
        return report