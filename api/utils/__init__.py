import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_API_URL = os.getenv('DATABASE_API_URL')