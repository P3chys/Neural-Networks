import logging
class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.setup_logging()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler()
            ]
        )

    def log_info(self, message):
        logging.info(message)

    def log_error(self, message):
        logging.error(message)

    def log_warning(self, message):
        logging.warning(message)

    def log_debug(self, message):
        logging.debug(message)

    def log_critical(self, message):
        logging.critical(message)

    def log_exception(self, message):
        logging.exception(message)