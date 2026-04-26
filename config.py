import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
    MONGO_URI = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/meddiagno"
    )
    UPLOAD_FOLDER    = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    WTF_CSRF_ENABLED = True