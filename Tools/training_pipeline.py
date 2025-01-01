import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
import os
from datetime import datetime
import logging

class StockTrainingPipeline:
    def __init__(self, model, log_dir="training_logs"):
        """
        Initialize the training pipeline with a model and logging configuration.
        """
        self.model = model
        self.log_dir = log_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for training progress."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        
    def create_callbacks(self, checkpoint_dir):
        """
        Create training callbacks with updated model saving format.
        
        Args:
            checkpoint_dir: Directory where model checkpoints will be saved
        """
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/tensorboard", exist_ok=True)
        
        # Create a unique model filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_path = os.path.join(checkpoint_dir, f'model_{timestamp}.keras')
        
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint with updated .keras format
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=f"{self.log_dir}/tensorboard/{timestamp}",
                histogram_freq=1
            )
        ]
        
        return callbacks, model_path
    
    def train(self, 
              X_train, y_train, 
              X_val, y_val,
              epochs=100,
              batch_size=32,
              checkpoint_dir='models'):
        """
        Execute the training pipeline with updated model saving.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save model checkpoints
        """
        logging.info("Starting training pipeline...")
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Validation data shape: {X_val.shape}")
        
        # Create callbacks and get model path
        callbacks, model_path = self.create_callbacks(checkpoint_dir)
        
        # Execute training
        try:
            history = self.model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            logging.info("Training completed successfully")
            logging.info(f"Model saved to: {model_path}")
            self._log_training_results(history)
            
            return history
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
            
    def _log_training_results(self, history):
        """Log final training metrics."""
        final_epoch = len(history.history['loss'])
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logging.info(f"Training completed after {final_epoch} epochs")
        logging.info(f"Final training loss: {final_loss:.4f}")
        logging.info(f"Final validation loss: {final_val_loss:.4f}")