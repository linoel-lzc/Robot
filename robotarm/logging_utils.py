import logging
import sys
import os
from pathlib import Path

def setup_logging(program_name: str, log_dir: str = "logs", level=logging.INFO, filename: str = None):
    """
    Setup logging configuration.
    
    Args:
        program_name: The name of the program (used as tag and filename).
        log_dir: Directory to save log files.
        level: Logging level.
        filename: Optional custom filename. If provided, overrides program_name based filename.
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Log filename based on program name or provided filename
    if filename:
        log_file = log_path / filename
    else:
        log_file = log_path / f"{program_name.lower()}.log"
    
    # Define format
    # We include the program name (tag) in the format string directly
    log_format = f'%(asctime)s - [{program_name}] - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")
