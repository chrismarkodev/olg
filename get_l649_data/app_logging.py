""" App logging configuration. """

import logging

import config

# set and configure parent logger
logger = logging.getLogger(config.APP_NAME_SHORT)
logger.setLevel(logging.INFO)
# define console logging with its own different logging level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
console_formatter = logging.Formatter('%(asctime)s %(filename)s %(message)s', '%H:%M:%S')
console_handler.setFormatter(console_formatter)
# define logfile logging with its own different logging level
logfile_handler = logging.FileHandler(filename=f"{config.APP_NAME}.log", mode='w')
logfile_handler.setLevel(logging.INFO)
logger.addHandler(logfile_handler)
logfile_formatter = logging.Formatter('%(asctime)s %(filename)s %(message)s', '%d/%m/%Y %H:%M:%S')
logfile_handler.setFormatter(logfile_formatter)

# set child logger and create started msg
def init_logging():
    """Initialize logging."""
    logger_child = logger.getChild(f"sub{__name__}")
    logger_child.info("Logging initialized")

if __name__ == "__main__":
    init_logging()