import logging
import sys


class NullLogger:
    """A logger that does nothing but can be passed around."""
    
    def debug(self, msg, *args, **kwargs):
        pass
    
    def info(self, msg, *args, **kwargs):
        pass
    
    def warning(self, msg, *args, **kwargs):
        pass
    
    def error(self, msg, *args, **kwargs):
        pass
    
    def critical(self, msg, *args, **kwargs):
        pass
    
    def exception(self, msg, *args, **kwargs):
        pass


def setup_null_logger() -> NullLogger:
    """Create a logger that does nothing."""
    return NullLogger()


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger for console output.

    Args:
        name (str): Logger name, defaults to module name
        level (int): Logging level, defaults to INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
