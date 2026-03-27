import requests
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class PushoverService:

    def send(self, message):
        logger.info(message)        
        if(os.getenv("IS_LOCAL", "FALSE").strip().upper() == "TRUE"):
            logger.info("IS_LOCAL is TRUE — skipping actual push notification")
            return
        
        requests.post(
            os.getenv("PUSHOVER_URL"),
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": message,
            }
        )