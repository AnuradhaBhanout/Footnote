
import os
from dotenv import load_dotenv,find_dotenv


_ = load_dotenv(find_dotenv())
DATABASE_URL = os.getenv("DATABASE_URL")