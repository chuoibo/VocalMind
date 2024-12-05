import pymongo
import logging

class BaseDatabase(object):
    def __init__(self, config):
        self.hostname = config['hostname']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
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
