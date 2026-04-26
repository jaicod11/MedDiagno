# 🏥 MedDiagno — AI Medical Diagnostics Platform

A full-stack Flask web application for AI-powered medical screening across **3 disease categories** with user authentication, **MongoDB Atlas** cloud database, personal dashboards, and prediction history.

## 🚀 Features

| Feature | Details |
|---|---|
| 🔐 Authentication | Register / login / logout with bcrypt-hashed passwords |
| 🩸 Diabetes Screening | XGBoost model · 8 clinical markers · **96.7% accuracy** |
| ❤️ Heart Disease | Gradient Boosting · 13 cardiovascular parameters · **88.7% accuracy** |
| 🔬 Skin Cancer | Image upload · feature extraction · **99.8% accuracy** |
| 📊 Dashboard | Weekly activity chart, prediction type distribution, recent history |
| 📋 History | Full filterable prediction log per user |
| 🍃 MongoDB Atlas | Cloud NoSQL database — no local DB setup required |
| 🐳 Docker Ready | One-command deployment |

## 🛠️ Tech Stack

- **Backend**: Flask, Flask-PyMongo, Flask-Login
- **Database**: MongoDB Atlas (cloud) via `pymongo[srv]`
- **ML**: XGBoost, scikit-learn (GradientBoosting), joblib
- **Frontend**: Vanilla JS, Chart.js, custom CSS design system
- **Deployment**: Docker + Gunicorn

## ⚙️ Environment Setup

Create a `.env` file in the project root:

```env
SECRET_KEY=your-random-secret-key-here
MONGO_URI=mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/meddiagno
FLASK_ENV=development
```

### Getting your MongoDB Atlas URI

1. Go to [cloud.mongodb.com](https://cloud.mongodb.com) and create a free **M0** cluster
2. Click **Connect** → **Drivers** → copy the connection string
3. Replace `<user>` and `<password>` with your Atlas credentials
4. In Atlas → **Network Access** → add `0.0.0.0/0` to allow connections

> MongoDB creates the `users` and `predictions` collections automatically on first insert — no migrations needed.

## ⚡ Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Add your .env file with MONGO_URI (see above)

# 3. Train models (first time only — .pkl files are included in the repo)
python train_models.py

# 4. Run
python app.py
# → http://localhost:5000
```

## 🐳 Docker

```bash
docker build -t meddiagno .
docker run --env-file .env -p 5000:5000 meddiagno
```

> Make sure your `.env` file contains a valid `MONGO_URI` before building.

## 📁 Project Structure

```
meddiagno/
├── app.py                  # Flask application factory
├── config.py               # Env-based config (MONGO_URI, SECRET_KEY, etc.)
├── extensions.py           # PyMongo + Flask-Login initialisation
├── train_models.py         # Trains and saves all 3 ML models
├── requirements.txt        # All pinned dependencies
├── Dockerfile              # Production container (Gunicorn)
├── .env                    # Local secrets — never commit this
├── .gitignore
│
├── models/                 # Pre-trained .pkl files
│   ├── diabetes_model.pkl  # XGBoost — 96.7% accuracy
│   ├── diabetes_scaler.pkl # StandardScaler for diabetes features
│   ├── heart_model.pkl     # GradientBoosting — 88.7% accuracy
│   ├── heart_scaler.pkl    # StandardScaler for heart features
│   ├── skin_model.pkl      # GradientBoosting — 99.8% accuracy
│   └── skin_scaler.pkl     # StandardScaler for image features
│
├── database/
│   ├── __init__.py
│   └── models.py           # User + Prediction classes (PyMongo, no ORM)
│
├── routes/
│   ├── __init__.py
│   ├── auth.py             # Login / register / logout
│   ├── dashboard.py        # Dashboard + profile views
│   └── predictions.py      # ML inference endpoints + history
│
├── templates/
│   ├── base.html           # Shared sidebar layout (all authenticated pages)
│   ├── landing.html        # Public marketing / landing page
│   ├── auth/
│   │   ├── login.html
│   │   └── register.html
│   ├── dashboard/
│   │   └── index.html      # Charts, stats, quick actions
│   ├── predictions/
│   │   ├── diabetes.html   # Slider-based input form
│   │   ├── heart.html      # Dropdown + slider form
│   │   ├── skin.html       # Drag-and-drop image upload
│   │   └── history.html    # Filterable prediction log
│   └── profile/
│       └── index.html      # User stats + recent activity
│
└── static/
    ├── css/main.css        # Full custom design system (500 lines)
    └── js/app.js           # Chart.js, async forms, image upload (192 lines)
```

## 🗄️ Database Schema (MongoDB)

### `users` collection
```json
{
  "_id":           "ObjectId",
  "name":          "Jane Doe",
  "email":         "jane@example.com",
  "password_hash": "bcrypt hash",
  "created_at":    "ISODate",
  "last_login":    "ISODate"
}
```

### `predictions` collection
```json
{
  "_id":             "ObjectId",
  "user_id":         "string (User ObjectId)",
  "prediction_type": "diabetes | heart | skin",
  "result":          "Not Diabetic | Diabetic | ...",
  "confidence":      96.7,
  "input_data":      { "glucose": 120, "bmi": 27.5, ... },
  "created_at":      "ISODate"
}
```

> `input_data` is stored as a **native MongoDB document** (not a JSON string), making it queryable directly from Atlas.

## 🤖 ML Models

| Model | Algorithm | Features | Accuracy |
|---|---|---|---|
| Diabetes | XGBoost | 8 clinical markers | 96.7% |
| Heart Disease | Gradient Boosting | 13 cardiovascular params | 88.7% |
| Skin Cancer | Gradient Boosting | 10 image statistics | 99.8% |

Each model ships as a pair: `*_model.pkl` (the classifier) + `*_scaler.pkl` (StandardScaler fitted on training data). Both are required at inference time — the scaler normalises raw inputs before the model sees them.

## ⚠️ Disclaimer

This is a **portfolio/educational project**. Models are trained on synthetic data. Do not use for real medical decisions.
