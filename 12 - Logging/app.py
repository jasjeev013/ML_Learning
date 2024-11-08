import logging

# Configure the logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler('app.log'),
                              logging.StreamHandler()]) 

# Create a logger
logger = logging.getLogger("ArthemiticApp")


def sum(a,b):
    logger.debug('Sum function called')
    return a+b

def subtract(a,b):
    logger.debug('Subtract function called')
    return a-b



# Example usage
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')