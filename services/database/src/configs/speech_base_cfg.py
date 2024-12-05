import os

from dotenv import load_dotenv

from src.utils import *
from src.utils.common import read_yaml
from src.utils.api_logger import logging

load_dotenv()

class Configuration():
    def __init__(self):
        self.config = read_yaml(DATABASE_CFG_FILE_PATH)
        self.BROKER_URI = os.getenv('MQ_URL')
        self.BACKEND_URI = os.getenv('REDIS_URL')
        logging.info('Init LLM database configuration parameters')
    
    def init_database(self):
        logging.info('Initialize LLM database')
        return self.config['speech_database']
    
