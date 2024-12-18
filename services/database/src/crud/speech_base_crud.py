from src.db.speech_db import SpeechDatabase
from src.utils.api_logger import logging

class SpeechCRUD:
    def __init__(self):
        self.database_instance = SpeechDatabase()
        logging.info('Initialize CRUD base process for SPeech database')
    

    def save_task_metadata(self, task_id, input_path, time_sent):
        logging.info("Save initial task metadata.")
        collection = self.database_instance.get_collection()
        
        collection.insert_one({
            "task_id": task_id,
            "status": "pending",
            "input_path": input_path,
            "output_path": None,
            "time_sent": time_sent
        })

    def update_task_metadata(self, task_id, status, output_path=None):
        logging.info("Update task status and output path.")

        collection = self.database_instance.get_collection()

        update_fields = {"status": status}
        if output_path:
            update_fields["output_path"] = output_path
        
        collection.update_one(
            {"task_id": task_id},
            {"$set": update_fields}
        )