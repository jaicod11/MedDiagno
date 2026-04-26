# 🏥 MedDiagno — AI Medical Diagnostics Platform

A full-stack Flask web application for AI-powered medical screening across **3 disease categories** with user authentication, personal dashboards, and prediction history.

## 🚀 Features

| Feature | Details |
|---|---|
| 🔐 Authentication | Register / login / logout with bcrypt-hashed passwords |
| 🩸 Diabetes Screening | XGBoost model · 8 clinical markers · **96.7% accuracy** |
| ❤️ Heart Disease | Gradient Boosting · 13 cardiovascular parameters · **88.7% accuracy** |
| 🔬 Skin Cancer | Image upload · feature extraction · **99.8% accuracy** |
| 📊 Dashboard | Weekly activity chart, prediction type distribution, recent history |
| 📋 History | Full filterable prediction log per user |
| 🐳 Docker Ready | One-command deployment |

## 🛠️ Tech Stack

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **ML**: XGBoost, scikit-learn (GradientBoosting), joblib
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Frontend**: Vanilla JS, Chart.js, custom CSS design system
- **Deployment**: Docker + Gunicorn

## ⚡ Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Train models (first time only)
python train_models.py

# 3. Run
python app.py
# → http://localhost:5000
```

## 🐳 Docker

```bash
docker build -t meddiagno .
docker run -p 5000:5000 meddiagno
```

## 📁 Project Structure

```
meddiagno/
├── app.py                  # Flask factory
├── config.py               # Configuration
├── extensions.py           # SQLAlchemy, LoginManager
├── train_models.py         # Model training script
├── models/                 # Trained .pkl files (auto-generated)
├── database/
│   └── models.py           # User + Prediction ORM models
├── routes/
│   ├── auth.py             # Login / register / logout
│   ├── dashboard.py        # Dashboard + profile
│   └── predictions.py      # ML inference endpoints
├── templates/
│   ├── base.html           # Sidebar layout
│   ├── landing.html        # Public landing page
│   ├── auth/               # Login & register
│   ├── dashboard/          # Main dashboard
│   ├── predictions/        # 3 prediction pages + history
│   └── profile/            # User profile
└── static/
    ├── css/main.css        # Full design system
    └── js/app.js           # Chart init, form handling, image upload
```

## ⚠️ Disclaimer

This is a **portfolio/educational project**. Models are trained on synthetic data. Do not use for real medical decisions.
