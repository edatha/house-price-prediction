import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import default_logger as logger
from src.models.model import ModelBuilder
from src.utils.config import config


class ModelTrainer:
    """Handles training, evaluation, and selection of the best model."""
    
    def __init__(self, X, y, full_pipeline):
        """
        Initialize ModelTrainer.
        
        Args:
            X (pd.DataFrame): Processed features.
            y (pd.Series): Target variable.
            pipeline (Pipeline): Preprocessing pipeline.
        """
        try:
            self.X = X
            self.y = y
            self.pipeline = full_pipeline
            self.models = ModelBuilder().get_models()
            self.best_model = None
            self.experiment_name = config.get('mlflow.experiment_name', 'house_price_experiment')
            self.artifact_path = config.get('mlflow.artifact_path', 'model')
            self.tracking_uri = config.get('mlflow.tracking_uri', 'sqlite:///mlflow.db')
            self._setup_mlflow()
            logger.info("ModelTrainer initialized.")
        
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise
    
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            # Check if the experiment already exists, if not, create it
            existing_experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if existing_experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow setup completed.")
        
        except Exception as e:
            logger.error(f"Error setting up MLflow: {e}")
            raise
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.
        
        Returns:
            Tuple: Splitted training and testing datasets.
        """
        try:
            logger.info("Splitting data into train and test sets...")
            return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        
        except Exception as e:
            logger.error(f"Error during train-test split: {e}")
            raise
    
    def train_and_evaluate(self):
        """
        Train models and evaluate them. Save the best model to 'artifacts/best_model.pkl'.
        """
        try:
            logger.info("Starting training and evaluation...")
            X_train, X_test, y_train, y_test = self.train_test_split()
            
            # Convert integer columns to float64
            X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include=['int']).columns})
            X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include=['int']).columns})
            
            best_score = float("inf")  # Initialize with a high value for RMSE
            for model_name, details in self.models.items():
                pipeline = self.pipeline
                pipeline.steps.append(('model', details['model']))
                logger.info(f"Training model: {model_name}")
                
                # Integrate preprocessing pipeline with the model
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=details['params'],
                    cv=2,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1
                )
                
                with mlflow.start_run(run_name=model_name, nested=True):
                    # Log parameters
                    mlflow.log_params(details['params'])
                    
                    # Fit and evaluate model
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.best_estimator_.predict(X_test)
                    
                    # Compute metrics
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    logger.info(f"Model: {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2
                    })
                    
                    # Get selected features after SelectKBest
                    try:
                        pipeline.steps.pop()  # Remove the model step temporarily
                        pipeline.fit(X_train, y_train)  # Fit pipeline to apply SelectKBest
                        selected_features = pipeline.named_steps['select_k_best'].get_support(indices=True)
                        selected_feature_names = np.array(X_train.columns)[selected_features]
                        input_example = X_train[selected_feature_names].head(1)
                        logger.info(f"Selected features for {model_name}: {selected_feature_names}")
                    except Exception as e:
                        logger.warning(f"Could not retrieve selected features for {model_name}: {e}")
                    
                    # Update the best model if current model is better
                    if rmse < best_score:
                        best_score = rmse
                        self.best_model = grid_search.best_estimator_
                        logger.info(f"New best model: {model_name} with RMSE: {rmse:.4f}")

                    # Ensure input example contains all necessary columns, even if some are missing
                    input_example = self._ensure_input_example(X_train, input_example)
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        sk_model=self.best_model,
                        artifact_path=self.artifact_path,
                        registered_model_name=f"{self.experiment_name}_{model_name}",
                        input_example=input_example
                    )
                    
            # Save the best model locally
            if self.best_model:
                joblib.dump(self.best_model, os.path.join(config.get('artifacts'), "best_model.pkl"))
                logger.info("Best model saved to disk.")
            
        except Exception as e:
            logger.error(f"Error during training and evaluation: {e}")
            raise
        
    def _ensure_input_example(self, X_train, input_example):
        """Ensure the input example contains all necessary columns."""
        # List of all columns expected by the model
        required_columns = X_train.columns
        missing_columns = required_columns.difference(input_example.columns)
        
        # Add missing columns with NaN values
        for column in missing_columns:
            input_example[column] = np.nan
        
        # Reorder columns to match the expected input format
        input_example = input_example[required_columns]
        
        return input_example