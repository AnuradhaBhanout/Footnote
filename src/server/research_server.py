import logging
import os


from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


from db.db import init_db


init_db()               # creates tables on first run, safe to call every time


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/research_server_debug.log"),
        logging.StreamHandler() # This also prints to your terminal
    ]
)




