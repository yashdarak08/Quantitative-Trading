"""
Logging utility for the Quantitative Trading System.
Provides standardized logging across the project.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    Set up a logger with file and/or console output.
    
    Parameters:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (int, optional): Logging level
        console_output (bool, optional): Whether to output to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_default_logger():
    """
    Get the default logger for the application.
    Creates logs directory and log file if they don't exist.
    
    Returns:
        logging.Logger: Default logger
    """
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"quant_trading_{timestamp}.log"
    
    # Set up and return logger
    return setup_logger('quant_trading', log_file=str(log_file))

# Default logger
logger = get_default_logger()