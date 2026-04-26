from extensions import mongo
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from datetime import datetime

class User(UserMixin):
    def __init__(self, data):
        self.id       = str(data["_id"])
        self.name     = data["name"]
        self.email    = data["email"]
        self.password_hash = data["password_hash"]
        self.created_at    = data.get("created_at", datetime.utcnow())
        self.last_login    = data.get("last_login")

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def initials(self):
        parts = self.name.strip().split()
        return (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else self.name[:2].upper()

    @staticmethod
    def create(name, email, password):
        doc = {
            "name": name, "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow()
        }
        result = mongo.db.users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return User(doc)

    @staticmethod
    def find_by_email(email):
        data = mongo.db.users.find_one({"email": email})
        return User(data) if data else None

    @staticmethod
    def find_by_id(user_id):
        data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
        return User(data) if data else None

    def prediction_counts(self):
        pipeline = [
            {"$match": {"user_id": self.id}},
            {"$group": {"_id": "$prediction_type", "count": {"$sum": 1}}}
        ]
        result = {d["_id"]: d["count"] for d in mongo.db.predictions.aggregate(pipeline)}
        return {"diabetes": result.get("diabetes", 0),
                "heart":    result.get("heart", 0),
                "skin":     result.get("skin", 0)}

    def recent_predictions(self, limit=5):
        cursor = mongo.db.predictions.find({"user_id": self.id}) \
                                      .sort("created_at", -1).limit(limit)
        return [Prediction(p) for p in cursor]


class Prediction:
    def __init__(self, data):
        self.id              = str(data["_id"])
        self.user_id         = data["user_id"]
        self.prediction_type = data["prediction_type"]
        self.result          = data["result"]
        self.confidence      = data["confidence"]
        self.created_at      = data.get("created_at", datetime.utcnow())

    @property
    def risk_level(self):
        return "low" if self.result in ["Not Diabetic", "No Heart Disease", "Benign"] else "high"

    @property
    def type_icon(self):
        return {"diabetes": "🩸", "heart": "❤️", "skin": "🔬"}.get(self.prediction_type, "🏥")

    @property
    def type_label(self):
        return {"diabetes": "Diabetes", "heart": "Heart Disease", "skin": "Skin Cancer"}.get(self.prediction_type, "")