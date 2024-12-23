from src.db.speech_db import SpeechDatabase

from src.utils.api_logger import logging
from fastapi import HTTPException, status

class SpeechCRUD:
    def __init__(self):
        self.database_instance = SpeechDatabase()
        logging.info('Initialize CRUD base process for Speech database')
    

    def get_tasks_for_user(self, user_id):
        logging.info("Get all tasks for specific user")
        collection = self.database_instance.get_collection()
        user_doc = collection.find_one({"user_id": user_id}, {"_id": 0, "tasks": 1})
        if not user_doc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
        return user_doc["tasks"]

    
    def get_task_metadata(self, task_id):
        logging.info("Get task metadata by task_id")
        collection = self.database_instance.get_collection()

        task_doc = collection.find_one(
            {"tasks.task_id": task_id},
            {"_id": 0, "tasks.$": 1}
        )

        if not task_doc or not task_doc.get("tasks"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Task not found")

        return task_doc["tasks"][0]
    

    def save_task_metadata(self, user_id, task_id, input_path_remote, time_sent):
        logging.info("Save initial task metadata.")
        collection = self.database_instance.get_collection()
        
        task = {
            "task_id": task_id,
            "status": "pending",
            "input_path_remote": input_path_remote,
            "input_path_local": None,
            "output_path": None,
            "time_sent": time_sent
        }
        
        collection.update_one(
            {"user_id": user_id},
            {"$push": {"tasks": task}},
            upsert=True  
        )


    def update_task_metadata(self, task_id, status, input_path_local, output_path=None):
        logging.info("Update task status and output path.")

        collection = self.database_instance.get_collection()

        update_fields = {}
        if status:
            update_fields["tasks.$.status"] = status
        if input_path_local:
            update_fields["tasks.$.input_path_local"] = input_path_local
        if output_path:
            update_fields["tasks.$.output_path"] = output_path

        result = collection.update_one(
            {"tasks.task_id": task_id}, 
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            logging.warning(f"Task with id {task_id} not found.")
            raise HTTPException(status_code=404, detail="Task not found")
        
        logging.info(f"Task {task_id} successfully updated.")
        return {"task_id": task_id, "updated_fields": update_fields}