import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import json
from PIL import Image
import io
from xgboost import XGBClassifier
import onnxruntime as rt
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- UPDATED FUNCTION ---
def load_diabetes_model():
    # Load the XGBoost model from the stable .json file
    model = XGBClassifier()
    model.load_model('models/diabetes_model.json')
    
    # Load the ONNX scaler as an "inference session"
    scaler_session = rt.InferenceSession('models/diabetes_scaler.onnx', providers=["CPUExecutionProvider"])
    
    print("Diabetes model (.json) and scaler (.onnx) loaded.")
    return model, scaler_session # Return the model and the ONNX session

# --- UPDATED FUNCTION ---
def load_heart_model():
    # Load the entire ONNX pipeline (scaler + model) as an "inference session"
    model_session = rt.InferenceSession('models/heart_model.onnx', providers=["CPUExecutionProvider"])
    
    print("Heart model pipeline (.onnx) loaded.")
    return model_session # Return the ONNX session

# --- UNCHANGED FUNCTION ---
def load_skin_cancer_model():
    # This was already in a stable format (.keras)
    model = load_model('models/skin_cancer_model.keras')
    with open('models/skin_class_indices.json', 'r') as file:
        class_indices = json.load(file)
    
    print("Skin cancer model (.keras) and indices (.json) loaded.")
    return model, class_indices

# --- UPDATED MODEL LOADING AT STARTUP ---
try:
    diabetes_model, diabetes_scaler_session = load_diabetes_model()
    heart_model_session = load_heart_model()
    skin_cancer_model, skin_class_indices = load_skin_cancer_model()
    print("\n--- All models loaded successfully! ---")

except Exception as e:
    print(f"--- FATAL ERROR: COULD NOT LOAD MODELS ---")
    print(f"Error: {e}")
    print("Please make sure all model files exist in the 'models' folder:")
    print(" - diabetes_model.json")
    print(" - diabetes_scaler.onnx")
    print(" - heart_model.onnx")
    print(" - skin_cancer_model.keras")
    print(" - skin_class_indices.json")
    print("\nAlso make sure you have run: pip install onnxruntime xgboost tensorflow")
    diabetes_model, diabetes_scaler_session = None, None
    heart_model_session = None
    skin_cancer_model, skin_class_indices = None, None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({'success': False, 'error': 'Diabetes model is not loaded. Check server logs.'})
    try:
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree']),
            float(request.form['age'])
        ]

        # 1. Convert data to float32 for ONNX
        data_array = np.array(data).reshape(1, -1).astype(np.float32)

        # 2. Use the ONNX scaler
        scaler_input_name = diabetes_scaler_session.get_inputs()[0].name
        scaler_output_name = diabetes_scaler_session.get_outputs()[0].name
        scaled_data = diabetes_scaler_session.run([scaler_output_name], {scaler_input_name: data_array})[0]

        # 3. Use the XGBoost model to predict on the scaled data
        prediction = diabetes_model.predict(scaled_data)
        probability = diabetes_model.predict_proba(scaled_data)
        
        # FIX: Convert numpy types to standard Python types
        pred_label = int(prediction[0])
        confidence = float(probability[0][pred_label]) * 100
        result = "Diabetic" if pred_label == 1 else "Not Diabetic"
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    if not heart_model_session:
        return jsonify({'success': False, 'error': 'Heart model is not loaded. Check server logs.'})
    try:
        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        # 1. Convert data to float32 for ONNX
        data_array = np.array(data).reshape(1, -1).astype(np.float32)

        # 2. Use the ONNX pipeline
        input_name = heart_model_session.get_inputs()[0].name
        output_label_name = heart_model_session.get_outputs()[0].name
        output_prob_name = heart_model_session.get_outputs()[1].name
        onnx_result = heart_model_session.run([output_label_name, output_prob_name], {input_name: data_array})
        
        # 3. Get results & FIX numpy types
        pred_label = int(onnx_result[0][0])
        probabilities = onnx_result[1][0]
        confidence = float(probabilities[pred_label]) * 100
        result = "Heart Disease Present" if pred_label == 1 else "No Heart Disease"
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skin_cancer.html')

@app.route('/predict_skin_cancer', methods=['POST'])
def predict_skin_cancer():
    if not skin_cancer_model:
        return jsonify({'success': False, 'error': 'Skin cancer model is not loaded. Check server logs.'})
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if file and allowed_file(file.filename):
            image = Image.open(io.BytesIO(file.read()))

            # --- *** THIS IS THE FIX (PART 2) *** ---
            # 1. Ensure image is 3-channel (RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 2. Resize
            image = image.resize((224, 224)) 
            
            # 3. Convert to numpy array
            image_array = np.array(image) 
            
            # 4. Add batch dimension FIRST
            image_batch = np.expand_dims(image_array, axis=0) 

            # 5. Use the official VGG16 preprocess_input function
            # This correctly handles mean subtraction and BGR conversion
            processed_image = preprocess_input(image_batch.astype(np.float32))

            # 6. Predict on the correctly processed image
            prediction_prob = float(skin_cancer_model.predict(processed_image)[0][0])
            
            # 7. Get class and confidence
            prediction = 1 if prediction_prob > 0.5 else 0
            confidence = (prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob) * 100

            # 8. Get the class name
            class_names = {v: k for k, v in skin_class_indices.items()}
            result = class_names[prediction]
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': round(confidence, 2)
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'})
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)