import pymongo
import os

from dotenv import load_dotenv

from src.utils.api_logger import logging

load_dotenv()

class BaseDatabase(object):
    def __init__(self, config):
        self.hostname = config['hostname']
        self.port = config['port']
        self.user = os.getenv('DATABASE_USER', 'guest')
        self.password = os.getenv('DATABASE_PASSWORD', 'guest')
        if (self.user == None or self.password == None) or (self.user == '' or self.password == ''):
            self.url = f'mongodb://{self.hostname}:{self.port}'
        else:
            self.url = f"mongodb://{self.user}:{self.password}@{self.hostname}:{self.port}"
                
        self.initialize()
    
    def initialize(self):
        try:
            self.client = pymongo.MongoClient(self.url)
            logging.info('Connected to MongoDB successfully !')
        except pymongo.errors.ServerSelectionTimeoutError as er:
            logging.info(f'Cannot connect to MongoDB: {er}')
