from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataProcessor
from src.utils.logger import default_logger as logger

if __name__ == "__main__":
    try: 
        logger.info("Initializing Class...")
        data_loader = DataLoader()
        data_processor = DataProcessor()
        
        logger.info("Starting Download dataset...")
        df = data_loader.load_data()
        logger.info("Data has been loaded")
        
        data_loader.validate_data(df)
        
        data_processor.process_data(df)
        
    except Exception as e:
        logger.error(f"Error to load the data {e}")
        raise