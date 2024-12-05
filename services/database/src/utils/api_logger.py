import logging

log_filename = "/speech/logs/api.log"

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  
        logging.StreamHandler() 
    ]
)


