import numpy as np
import pandas as pd
from scipy import stats
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
        """
        # Get model predictions 
        y_pred = self.model.predict(X_test)
        
        try:
            # Store original shapes
            original_shape = y_test.shape
            
            # Reshape predictions and test values into 2D arrays
            y_test_2d = y_test.reshape(-1, 1)
            y_pred_2d = y_pred.reshape(-1, 1)
            
            # Inverse transform
            y_test_orig = self.preprocessor.inverse_transform_predictions(y_test_2d, ticker)
            y_pred_orig = self.preprocessor.inverse_transform_predictions(y_pred_2d, ticker)
            
            # Reshape back to original shape
            y_test_orig = y_test_orig.reshape(original_shape)
            y_pred_orig = y_pred_orig.reshape(original_shape)
            
            # Calculate metrics using non-zero values to avoid division by zero
            metrics = {
                'mse': mean_squared_error(y_test_orig, y_pred_orig),
                'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
                'mae': mean_absolute_error(y_test_orig, y_pred_orig),
                'r2': r2_score(y_test_orig.flatten(), y_pred_orig.flatten())
            }
            
            # Calculate MAPE avoiding division by zero
            mask = y_test_orig != 0
            if mask.any():
                mape = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100
            else:
                mape = np.nan
            metrics['mape'] = mape
            
            # Calculate directional accuracy
            direction_test = np.diff(y_test_orig, axis=0)
            direction_pred = np.diff(y_pred_orig, axis=0)
            correct_direction = np.sum(np.sign(direction_test) == np.sign(direction_pred))
            total_directions = direction_test.size
            metrics['directional_accuracy'] = (correct_direction / total_directions) * 100
            
            self.evaluation_metrics[ticker] = metrics
            
            # Store predictions for plotting
            self.last_test_pred = {
                'y_test': y_test_orig,
                'y_pred': y_pred_orig
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {}
    
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
        if not hasattr(self, 'last_test_pred'):
            _ = self.evaluate_predictions(X_test, y_test, ticker)
            
        y_test_orig = self.last_test_pred['y_test']
        y_pred_orig = self.last_test_pred['y_pred']
        
        # Create time index for plotting
        time_index = np.arange(min(samples, len(y_test_orig)))
        
        # Calculate prediction errors for confidence intervals
        pred_errors = np.std(y_pred_orig - y_test_orig, axis=0)
        
        # Create subplot for each prediction day
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for day in range(self.prediction_horizon):
            ax = axes[day]
            
            # Plot actual values
            ax.plot(time_index, y_test_orig[:samples, day], 
                   label='Actual', color='blue', alpha=0.7)
            
            # Plot predicted values
            ax.plot(time_index, y_pred_orig[:samples, day], 
                   label='Predicted', color='red', alpha=0.7)
            
            # Add confidence intervals
            ax.fill_between(time_index,
                          y_pred_orig[:samples, day] - 1.96 * pred_errors[day],
                          y_pred_orig[:samples, day] + 1.96 * pred_errors[day],
                          color='red', alpha=0.1)
            
            ax.set_title(f'Day {day + 1} Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.legend()
            
            # Add error metrics to plot
            mse = mean_squared_error(y_test_orig[:, day], y_pred_orig[:, day])
            mae = mean_absolute_error(y_test_orig[:, day], y_pred_orig[:, day])
            ax.text(0.05, 0.95, f'MSE: {mse:.2f}\nMAE: {mae:.2f}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def plot_error_distribution(self, 
                              X_test: np.ndarray, 
                              y_test: np.ndarray, 
                              ticker: str):
        """
        Plot the distribution of prediction errors.
        
        Args:
            X_test: Test features
            y_test: Actual test values
            ticker: Stock ticker symbol
        """
        if not hasattr(self, 'last_test_pred'):
            _ = self.evaluate_predictions(X_test, y_test, ticker)
            
        y_test_orig = self.last_test_pred['y_test']
        y_pred_orig = self.last_test_pred['y_pred']
        
        # Calculate errors
        errors = y_pred_orig - y_test_orig
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Error distribution histogram
        sns.histplot(errors.flatten(), kde=True, ax=ax1)
        ax1.set_title('Distribution of Prediction Errors')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Count')
        
        # Add distribution statistics
        stats_text = f'Mean: {np.mean(errors):.2f}\n'
        stats_text += f'Std: {np.std(errors):.2f}\n'
        stats_text += f'Skew: {stats.skew(errors.flatten()):.2f}\n'
        stats_text += f'Kurtosis: {stats.kurtosis(errors.flatten()):.2f}'
        ax1.text(0.95, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Q-Q plot
        stats.probplot(errors.flatten(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Prediction Errors')
        
        plt.tight_layout()
        plt.show()
        
        # Plot error by prediction horizon
        plt.figure(figsize=(10, 5))
        box_data = [errors[:, i] for i in range(self.prediction_horizon)]
        plt.boxplot(box_data, labels=[f'Day {i+1}' for i in range(self.prediction_horizon)])
        plt.title('Prediction Errors by Forecast Horizon')
        plt.xlabel('Prediction Day')
        plt.ylabel('Error')
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
        
        # Create detailed report DataFrame
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
            ],
            'Description': [
                'Average squared difference between predicted and actual values',
                'Square root of MSE, in original scale',
                'Average absolute difference between predicted and actual values',
                'Proportion of variance in target explained by model',
                'Percentage error relative to actual values',
                'Percentage of correct price movement direction predictions'
            ]
        })
        
        return report