from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Dict
from src.utils.logger import default_logger as logger
from src.utils.config import config

class ModelBuilder:
    """Defines models and hyperparameters."""
    
    def __init__(self):
        logger.info("ModelBuilder initialized.")
    
    def get_models(self) -> Dict[str, Dict]:
        """
        Defines models and hyperparameter grid for each model from config file.
        
        Returns:
            Dict[str, Dict]: Dictionary containing models and their parameter grids.
        """
        try:
            # Take the hyperparameters' models from config.ymal
            model_params = config.get("model_params", {})
            
            # Define the model and its hyperparameters
            models = {
                "RandomForestRegressor": {
                    "model": RandomForestRegressor(random_state=config.get('RANDOM_STATE')),
                    "params": model_params.get("RandomForestRegressor", {})
                },
                "SVR": {
                    "model": SVR(),
                    "params": model_params.get("SVR", {})
                }
            }
            
            logger.info("Models and hyperparameters loaded from config.")
            return models
        
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise
        
        except Exception as e:
            logger.error(f"Error in model configuration: {e}")
            raise