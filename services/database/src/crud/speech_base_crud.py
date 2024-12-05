import logging
import datetime

from fastapi import status, HTTPException
from copy import deepcopy

from src.db.speech_db import SpeechDatabase
from utils.api_logger import logging


class SpeechCRUD:
    def __init__(self):
        self.database_instance = SpeechDatabase()
        logging.info('Initialize CRUD base process for SPeech database')


    def log_user_interaction(self, user_id: str, query: str, answer: str):
        """Logs a user interaction with the app."""

        collection = self.database_instance.get_collection()
        if self.database_instance.check_text_by_id(id):
            raise HTTPException(status.HTTP_409_CONFLICT)
        
        interaction = {
            "user_id": user_id,
            "query": query,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()  # Store time in UTC format
        }
        result = collection.insert_one(interaction)
        return result.inserted_id


    def get_user_interactions(self, user_id: str):
        """Retrieve all interactions for a specific user."""

        collection = self.database_instance.get_collection()
        if self.database_instance.check_text_by_id(id):
            raise HTTPException(status.HTTP_409_CONFLICT)
        
        interactions = collection.find({"user_id": user_id}).sort("timestamp", -1)
        return list(interactions)