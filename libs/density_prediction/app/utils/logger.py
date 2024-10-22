import logging
import sys
from logging.handlers import TimedRotatingFileHandler

# Set up the logger
def setup_logger():
    logger = logging.getLogger("density_prediction")
    logger.setLevel(logging.INFO)

    # Create a console handler for output to the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Create a file handler for output to a log file (rotating logs daily)
    file_handler = TimedRotatingFileHandler('logs/inference.log', when='midnight', backupCount=7)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logger()
