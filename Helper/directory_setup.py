from config import Config
import os

class DirectorySetup:
    def setup_directories():
        """Create necessary directories if they don't exist."""
        directories = [
            'Data/',
            'Models/models/',
            Config.RAW_DATA_PATH,
            'Data/training_logs/',
            'Data/training_logs/tensorboard/'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)