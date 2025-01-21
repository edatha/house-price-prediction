from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataProcessor
from src.models.trainer import ModelTrainer
from src.utils.config import config
from src.utils.logger import default_logger as logger

def main():
    try:
        # Initialize DataLoader
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader(data_path=config.get('data_path'))
        
        # Load data
        df = data_loader.load_data()
        
        # Validate data
        if not data_loader.validate_data(df):
            logger.error("Data validation failed. Exiting...")
            return
        
        # Initialize DataProcessor
        logger.info("Initializing DataProcessor...")
        data_processor = DataProcessor(drop_columns=config.get('columns_to_drop', []))
        
        # Process data (feature selection and split)
        X, y, full_pipeline = data_processor.process_data(df)
        
        # Initialize ModelTrainer
        logger.info("Initializing ModelTrainer...")
        model_trainer = ModelTrainer(X, y, full_pipeline)
        
        # Train and evaluate model
        model_trainer.train_and_evaluate()

        logger.info("Pipeline run completed successfully.")
    
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()