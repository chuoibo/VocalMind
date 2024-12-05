from src.main_database import BaseDatabase
from src.configs.speech_base_cfg import Configuration

class SpeechDatabase(BaseDatabase):
    def __init__(self):
        self.config = Configuration().init_database()
        super(SpeechDatabase, self).__init__(self.config)

        database_name = self.config['database_name']
        collection_name = self.config['collection_name']
        self.database = self.client[database_name]
        self.base_collection = self.database[collection_name]
    
    
    def get_collection(self):
        return self.base_collection


    def check_text_by_id(self, text_id: str) -> bool:
        text_doc = self.base_collection.find_one({'id': text_id}, {'_id': 0})
        if text_doc is None:
            return False
        return True
