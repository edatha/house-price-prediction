from src.utils.config import config
from src.utils.logger import default_logger as logger
from src.models import model

if __name__ == "__main__":
    try:
        logger.info("Initializing the class...")
        model = model.ModelBuilder()
        models = model.get_models()
        
    except Exception as e:
        logger.error("Error to build the model {e}")
        raise