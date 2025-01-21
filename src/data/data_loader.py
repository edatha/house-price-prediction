import pandas as pd

from pathlib import Path
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataLoader:
     """Utilities for Loading the data"""
     
     def __init__(self, data_path:Optional[str] = None):
        """Initialize the class

        Args:
            data_path (Optional[str], optional): A path to data file. Defaults to None.
        """
        self.data_path = data_path or config.get('data_path')
        logger.info(f"Initialized DataLoader with Path: {self.data_path}")
         
     def load_data(self) -> pd.DataFrame:
        """
        Load data from file
        
        Returns:
            pd.DataFrame: a file typing DataFrame
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"The data was loaded successfully with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error to load the data: {e}")
            raise
        
     def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the data

        Args:
             df (pd.DataFrame): The data

        Returns:
             bool: True if validation passes, False otherwise
        """
        try:
            logger.info("Validating data...")
            
            # Check required columns
            required_columns = config.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            
            logger.info("Data validation completed")
            return True
        
        except Exception as e:
            logger.error(f"Error in validating data: {e}")
            return False