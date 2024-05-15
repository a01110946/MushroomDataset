# logging_config.py
"""
This module sets up the logging configuration for the project.
It configures the root logger, handlers, and formatters for logging messages.
"""

import logging
import sys

def setup_logging():
    # Set up the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler('pipeline.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)