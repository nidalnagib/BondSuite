import logging
from pathlib import Path
import os
from datetime import datetime

def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Use a fixed log file name
    log_file = log_dir / "bondalloc.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Use mode='a' to append to the log file instead of overwriting
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set level for root logger
    logger.setLevel(logging.INFO)
    
    return logger
