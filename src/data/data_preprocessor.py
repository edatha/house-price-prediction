import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from typing import Tuple, List
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataProcessor:
    """Handles data preprocessing and feature selection."""
    
    def __init__(self, drop_columns: List[str] = None, target: str = 'SalePrice', num_features: int = 10):
        """
        Args:
            drop_columns (List[str]): Columns to drop.
            target (str): Target column name.
            num_features (int): Number of features to select.
        """
        self.drop_columns = drop_columns if drop_columns is not None else config.get('columns_to_drop', [])
        self.target = target
        self.num_features = num_features
        self.preprocessor = None
        logger.info("DataProcessor initialized.")
    
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
        """
        Process the dataset by dropping columns, splitting features and target, creating transformers, and applying SelectKBest.
        
        Args:
            df (pd.DataFrame): Raw dataframe.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, Pipeline]: Processed features, target, and full pipeline.
        """
        logger.info("Starting data processing...")
        
        try:
            # Drop columns
            logger.info("Dropping some columns")
            df = df.drop(columns=self.drop_columns, errors='ignore')
            
            # Split features and target
            logger.info("Splitting our dataset into X and y")
            X = df.drop(columns=[self.target])
            y = df[self.target]
            
            # Identify numerical and categorical columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            # Define transformers
            logger.info("Creating a processor pipeline")
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Create preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
            
            # Create full pipeline with feature selection after preprocessing
            full_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('select_k_best', SelectKBest(mutual_info_regression, k=self.num_features))
            ])
            
            logger.info("Data processing completed.")
            return X, y, full_pipeline
        
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise  # Re-raise the exception to handle it further upstream