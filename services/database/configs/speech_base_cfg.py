import os

from dotenv import load_dotenv

from utils import *
from utils.common import read_yaml
from api.utils.api_logger import logging

load_dotenv()

class Configuration():
    def __init__(self):
        self.config = read_yaml(DATABASE_CFG_FILE_PATH)
        logging.info('Init Speech database configuration parameters')
    
    def init_database(self):
        logging.info('Initialize LLM database')
        return self.config['task_result']
    
