from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from extensions import db
from database.models import Prediction
import numpy as np
import joblib
import os
import json
import io
from PIL import Image
from datetime import datetime

pred_bp = Blueprint("predictions", __name__, url_prefix="/predict")

# ── lazy model loader ──────────────────────────────────────────────────────────
_cache = {}

def _load(name):
    if name not in _cache:
        path = os.path.join(current_app.config["MODEL_DIR"], name)
        _cache[name] = joblib.load(path)
    return _cache[name]

def _save_prediction(ptype, result, confidence, input_dict):
    from extensions import mongo
    mongo.db.predictions.insert_one({
        "user_id":         current_user.id,
        "prediction_type": ptype,
        "result":          result,
        "confidence":      round(confidence, 2),
        "input_data":      input_dict,
        "created_at":      datetime.utcnow()
    })

# ── Pages ──────────────────────────────────────────────────────────────────────
@pred_bp.route("/diabetes")
@login_required
def diabetes_page():
    return render_template("predictions/diabetes.html")

@pred_bp.route("/heart")
@login_required
def heart_page():
    return render_template("predictions/heart.html")

@pred_bp.route("/skin")
@login_required
def skin_page():
    return render_template("predictions/skin.html")

@pred_bp.route("/history")
@login_required
def history():
    ptype = request.args.get("type", "all")
    query = Prediction.query.filter_by(user_id=current_user.id)
    if ptype in ("diabetes", "heart", "skin"):
        query = query.filter_by(prediction_type=ptype)
    preds = query.order_by(Prediction.created_at.desc()).all()
    return render_template("predictions/history.html", predictions=preds, filter=ptype)

# ── API endpoints ──────────────────────────────────────────────────────────────
@pred_bp.route("/api/diabetes", methods=["POST"])
@login_required
def api_diabetes():
    try:
        fields = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                  "insulin", "bmi", "diabetes_pedigree", "age"]
        data = {}
        for f in fields:
            val = request.form.get(f)
            if val is None:
                return jsonify(success=False, error=f"Missing field: {f}")
            data[f] = float(val)

        # Validate ranges
        if not (0 <= data["glucose"] <= 300):
            return jsonify(success=False, error="Glucose must be between 0 and 300 mg/dL.")
        if not (10 <= data["bmi"] <= 70):
            return jsonify(success=False, error="BMI must be between 10 and 70.")

        scaler = _load("diabetes_scaler.pkl")
        model  = _load("diabetes_model.pkl")
        X = np.array([[data[f] for f in fields]], dtype=np.float32)
        X_s = scaler.transform(X)
        pred   = int(model.predict(X_s)[0])
        proba  = model.predict_proba(X_s)[0]
        conf   = float(proba[pred]) * 100
        result = "Diabetic" if pred == 1 else "Not Diabetic"

        _save_prediction("diabetes", result, conf, data)
        return jsonify(success=True, result=result, confidence=round(conf, 1),
                       risk="high" if pred == 1 else "low")
    except Exception as e:
        return jsonify(success=False, error=str(e))

@pred_bp.route("/api/heart", methods=["POST"])
@login_required
def api_heart():
    try:
        fields = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                  "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        data = {}
        for f in fields:
            val = request.form.get(f)
            if val is None:
                return jsonify(success=False, error=f"Missing field: {f}")
            data[f] = float(val)

        scaler = _load("heart_scaler.pkl")
        model  = _load("heart_model.pkl")
        X = np.array([[data[f] for f in fields]], dtype=np.float32)
        X_s = scaler.transform(X)
        pred   = int(model.predict(X_s)[0])
        proba  = model.predict_proba(X_s)[0]
        conf   = float(proba[pred]) * 100
        result = "Heart Disease Present" if pred == 1 else "No Heart Disease"

        _save_prediction("heart", result, conf, data)
        return jsonify(success=True, result=result, confidence=round(conf, 1),
                       risk="high" if pred == 1 else "low")
    except Exception as e:
        return jsonify(success=False, error=str(e))

@pred_bp.route("/api/skin", methods=["POST"])
@login_required
def api_skin():
    try:
        if "file" not in request.files:
            return jsonify(success=False, error="No file uploaded.")
        file = request.files["file"]
        if file.filename == "":
            return jsonify(success=False, error="No file selected.")
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in current_app.config["ALLOWED_EXTENSIONS"]:
            return jsonify(success=False, error="Only PNG/JPG images are accepted.")

        # Read image and extract features
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_small = img.resize((128, 128))
        arr = np.array(img_small, dtype=np.float32)

        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
        std_r,  std_g,  std_b  = r.std(),  g.std(),  b.std()

        # Dark pixel ratio (melanin proxy)
        luminance  = 0.299*r + 0.587*g + 0.114*b
        dark_ratio = (luminance < 100).mean()

        # Asymmetry: compare left vs right halves
        left  = arr[:, :64, :]
        right = arr[:, 64:, :]
        asymmetry = float(np.abs(left.mean(axis=(0,1)) - right.mean(axis=(0,1))).mean() / 255.0)

        # Color variance
        color_variance = float(arr.var())

        # Border irregularity (edge variance in center crop)
        center = arr[32:96, 32:96, :]
        border_irreg = float(np.gradient(center[:,:,0]).pop().std() / 255.0)

        features = [mean_r, mean_g, mean_b, std_r, std_g, std_b,
                    dark_ratio, asymmetry, color_variance, border_irreg]

        scaler = _load("skin_scaler.pkl")
        model  = _load("skin_model.pkl")
        X   = np.array([features], dtype=np.float32)
        X_s = scaler.transform(X)
        pred    = int(model.predict(X_s)[0])
        proba   = model.predict_proba(X_s)[0]
        conf    = float(proba[pred]) * 100
        result  = "Malignant (Melanoma)" if pred == 1 else "Benign"

        input_dict = {
            "mean_rgb": f"({mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f})",
            "dark_ratio": f"{dark_ratio:.2%}",
            "asymmetry":  f"{asymmetry:.3f}",
            "filename":   file.filename
        }
        _save_prediction("skin", result, conf, input_dict)
        return jsonify(success=True, result=result, confidence=round(conf, 1),
                       risk="high" if pred == 1 else "low")
    except Exception as e:
        return jsonify(success=False, error=str(e))
