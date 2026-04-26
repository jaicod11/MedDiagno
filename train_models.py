"""
MedDiagno - Model Training Script
Trains and saves all ML models to the /models directory.
Run this once: python train_models.py
"""
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

np.random.seed(42)
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
# 1. DIABETES MODEL  (XGBoost + StandardScaler)
# ─────────────────────────────────────────────
def make_diabetes_data(n=3000):
    X, y = [], []
    for _ in range(n):
        diab = np.random.random() < 0.35
        if diab:
            preg  = max(0, int(np.random.normal(4, 3)))
            gluc  = np.clip(np.random.normal(145, 35), 70, 280)
            bp    = np.clip(np.random.normal(78, 16), 40, 120)
            skin  = np.clip(np.random.normal(32, 11), 0, 80)
            ins   = np.clip(np.random.normal(160, 90), 0, 500)
            bmi   = np.clip(np.random.normal(36, 8), 15, 65)
            dpf   = np.clip(np.random.normal(0.65, 0.35), 0.05, 2.5)
            age   = np.clip(np.random.normal(46, 12), 21, 80)
        else:
            preg  = max(0, int(np.random.normal(2, 2)))
            gluc  = np.clip(np.random.normal(100, 20), 50, 160)
            bp    = np.clip(np.random.normal(68, 12), 40, 110)
            skin  = np.clip(np.random.normal(22, 7), 0, 60)
            ins   = np.clip(np.random.normal(75, 45), 0, 300)
            bmi   = np.clip(np.random.normal(27, 5), 15, 50)
            dpf   = np.clip(np.random.normal(0.35, 0.20), 0.05, 1.5)
            age   = np.clip(np.random.normal(33, 9), 21, 70)
        X.append([preg, gluc, bp, skin, ins, bmi, dpf, age])
        y.append(int(diab))
    return np.array(X, dtype=np.float32), np.array(y)

print("Training Diabetes model …")
Xd, yd = make_diabetes_data(3000)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42)
scaler_d = StandardScaler()
Xd_tr_s  = scaler_d.fit_transform(Xd_tr)
Xd_te_s  = scaler_d.transform(Xd_te)
model_d  = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                         subsample=0.85, colsample_bytree=0.85,
                         use_label_encoder=False, eval_metric="logloss", random_state=42)
model_d.fit(Xd_tr_s, yd_tr, eval_set=[(Xd_te_s, yd_te)], verbose=False)
acc_d = accuracy_score(yd_te, model_d.predict(Xd_te_s))
print(f"  Accuracy: {acc_d:.3f}")
joblib.dump(model_d,  "models/diabetes_model.pkl")
joblib.dump(scaler_d, "models/diabetes_scaler.pkl")
print("  Saved diabetes_model.pkl + diabetes_scaler.pkl")

# ─────────────────────────────────────────────
# 2. HEART DISEASE MODEL  (GradientBoosting)
# ─────────────────────────────────────────────
def make_heart_data(n=3000):
    X, y = [], []
    for _ in range(n):
        sick = np.random.random() < 0.45
        if sick:
            age     = np.clip(np.random.normal(58, 9), 29, 77)
            sex     = int(np.random.random() < 0.75)
            cp      = int(np.random.choice([0,1,2,3], p=[0.50,0.22,0.17,0.11]))
            trestbps= np.clip(np.random.normal(136, 20), 90, 200)
            chol    = np.clip(np.random.normal(258, 52), 130, 420)
            fbs     = int(np.random.random() < 0.20)
            restecg = int(np.random.choice([0,1,2], p=[0.45,0.45,0.10]))
            thalach = np.clip(np.random.normal(142, 25), 70, 202)
            exang   = int(np.random.random() < 0.55)
            oldpeak = np.clip(np.random.normal(1.6, 1.3), 0, 6.2)
            slope   = int(np.random.choice([0,1,2], p=[0.40,0.40,0.20]))
            ca      = int(np.random.choice([0,1,2,3], p=[0.35,0.32,0.22,0.11]))
            thal    = int(np.random.choice([0,1,2,3], p=[0.05,0.05,0.40,0.50]))
        else:
            age     = np.clip(np.random.normal(52, 9), 29, 77)
            sex     = int(np.random.random() < 0.55)
            cp      = int(np.random.choice([0,1,2,3], p=[0.20,0.30,0.30,0.20]))
            trestbps= np.clip(np.random.normal(129, 17), 90, 180)
            chol    = np.clip(np.random.normal(240, 48), 130, 400)
            fbs     = int(np.random.random() < 0.12)
            restecg = int(np.random.choice([0,1,2], p=[0.55,0.40,0.05]))
            thalach = np.clip(np.random.normal(162, 22), 100, 202)
            exang   = int(np.random.random() < 0.20)
            oldpeak = np.clip(np.random.normal(0.6, 0.8), 0, 5)
            slope   = int(np.random.choice([0,1,2], p=[0.20,0.40,0.40]))
            ca      = int(np.random.choice([0,1,2,3], p=[0.60,0.25,0.10,0.05]))
            thal    = int(np.random.choice([0,1,2,3], p=[0.05,0.05,0.75,0.15]))
        X.append([age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal])
        y.append(int(sick))
    return np.array(X, dtype=np.float32), np.array(y)

print("Training Heart Disease model …")
Xh, yh = make_heart_data(3000)
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(Xh, yh, test_size=0.2, random_state=42)
scaler_h = StandardScaler()
Xh_tr_s  = scaler_h.fit_transform(Xh_tr)
Xh_te_s  = scaler_h.transform(Xh_te)
model_h  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.08, subsample=0.85, random_state=42)
model_h.fit(Xh_tr_s, yh_tr)
acc_h = accuracy_score(yh_te, model_h.predict(Xh_te_s))
print(f"  Accuracy: {acc_h:.3f}")
joblib.dump(model_h,  "models/heart_model.pkl")
joblib.dump(scaler_h, "models/heart_scaler.pkl")
print("  Saved heart_model.pkl + heart_scaler.pkl")

# ─────────────────────────────────────────────
# 3. SKIN CANCER MODEL  (feature-based GBM)
# ─────────────────────────────────────────────
# We extract color & texture statistics from images.
# This synthetic model is trained on those feature distributions.
def make_skin_data(n=3000):
    """
    Features (10): mean_r, mean_g, mean_b, std_r, std_g, std_b,
                   dark_ratio, asymmetry, color_variance, border_irreg
    """
    X, y = [], []
    for _ in range(n):
        malign = np.random.random() < 0.40
        if malign:
            mr  = np.clip(np.random.normal(130, 35), 50, 230)
            mg  = np.clip(np.random.normal(100, 30), 30, 200)
            mb  = np.clip(np.random.normal(95,  30), 30, 190)
            sr  = np.clip(np.random.normal(55, 18), 10, 100)
            sg  = np.clip(np.random.normal(48, 16), 10, 90)
            sb  = np.clip(np.random.normal(45, 15), 10, 90)
            dr  = np.clip(np.random.normal(0.42, 0.18), 0, 1)
            asy = np.clip(np.random.normal(0.38, 0.15), 0, 1)
            cv  = np.clip(np.random.normal(2800, 900), 500, 6000)
            bi  = np.clip(np.random.normal(0.60, 0.20), 0, 1)
        else:
            mr  = np.clip(np.random.normal(190, 30), 100, 255)
            mg  = np.clip(np.random.normal(155, 30), 80, 230)
            mb  = np.clip(np.random.normal(148, 28), 80, 220)
            sr  = np.clip(np.random.normal(32, 14), 5, 80)
            sg  = np.clip(np.random.normal(28, 12), 5, 70)
            sb  = np.clip(np.random.normal(26, 11), 5, 65)
            dr  = np.clip(np.random.normal(0.18, 0.12), 0, 0.6)
            asy = np.clip(np.random.normal(0.15, 0.10), 0, 0.5)
            cv  = np.clip(np.random.normal(1200, 600), 100, 4000)
            bi  = np.clip(np.random.normal(0.25, 0.15), 0, 0.7)
        X.append([mr, mg, mb, sr, sg, sb, dr, asy, cv, bi])
        y.append(int(malign))
    return np.array(X, dtype=np.float32), np.array(y)

print("Training Skin Cancer model …")
Xs, ys = make_skin_data(3000)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=42)
scaler_s = StandardScaler()
Xs_tr_s  = scaler_s.fit_transform(Xs_tr)
Xs_te_s  = scaler_s.transform(Xs_te)
model_s  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.08, subsample=0.85, random_state=42)
model_s.fit(Xs_tr_s, ys_tr)
acc_s = accuracy_score(ys_te, model_s.predict(Xs_te_s))
print(f"  Accuracy: {acc_s:.3f}")
joblib.dump(model_s,  "models/skin_model.pkl")
joblib.dump(scaler_s, "models/skin_scaler.pkl")
print("  Saved skin_model.pkl + skin_scaler.pkl")

print("\n✅ All models trained and saved successfully!")
print(f"  Diabetes  : {acc_d:.1%}")
print(f"  Heart     : {acc_h:.1%}")
print(f"  Skin      : {acc_s:.1%}")
