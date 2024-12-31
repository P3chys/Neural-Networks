import pandas as pd
class DateValidator:
    @staticmethod
    def validate_date_format(start_date: str, end_date: str):
        """
        Validate the date format for start and end dates.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
        
        Raises:
            ValueError: If the date format is invalid
        """
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            return start_date, end_date
        except ValueError as e:
            raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format. Error: {e}")
