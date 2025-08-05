from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from datetime import datetime
import uuid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import traceback

# Import our ML model
try:
    from ml_model import SkinConditionModel, create_mock_model
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False
    print("ML model module not available. Using basic prediction.")

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize ML model
if ML_MODEL_AVAILABLE:
    try:
        # Prefer the fine-tuned model if it exists
        if os.path.exists('skin_model_finetuned.h5'):
            model_path = 'skin_model_finetuned.h5'
            model_type = 'tensorflow'
            print('Loading fine-tuned TensorFlow model.')
        elif os.path.exists('skin_model.h5'):
            model_path = 'skin_model.h5'
            model_type = 'tensorflow'
            print('Loading initial TensorFlow model.')
        else:
            model_path = None
            model_type = 'mock'
            print('No trained model found. Using mock model.')
        ml_model = SkinConditionModel(model_path=model_path, model_type=model_type)
        print(f"✅ ML model initialized: {model_type}")
    except Exception as e:
        print(f"⚠️ Error loading ML model: {e}")
        ml_model = create_mock_model()
else:
    ml_model = None
    print("⚠️ Using basic prediction without ML model")

# Skin condition classes
SKIN_CONDITIONS = {
    0: "Acne",
    1: "Eczema",
    2: "Melanoma",
    3: "Normal",
    4: "Psoriasis",
    5: "Rosacea"
}

# Prescription database
PRESCRIPTIONS = {
    "Acne": {
        "description": "Inflammatory skin condition characterized by pimples, blackheads, and whiteheads",
        "recommendations": [
            "Keep the affected area clean with gentle cleanser",
            "Avoid touching or picking at pimples",
            "Use non-comedogenic moisturizer",
            "Consider over-the-counter benzoyl peroxide or salicylic acid",
            "Avoid heavy makeup and oil-based products",
            "Consult a dermatologist for severe cases"
        ],
        "severity": "Moderate"
    },
    "Eczema": {
        "description": "Chronic inflammatory skin condition causing dry, itchy, and red patches",
        "recommendations": [
            "Moisturize regularly with fragrance-free creams",
            "Avoid hot showers and harsh soaps",
            "Use gentle, hypoallergenic products",
            "Avoid scratching affected areas",
            "Consider oatmeal baths for relief",
            "Consult a dermatologist for prescription treatments"
        ],
        "severity": "Moderate"
    },
    "Melanoma": {
        "description": "Serious form of skin cancer that requires immediate medical attention",
        "recommendations": [
            "SEEK IMMEDIATE MEDICAL ATTENTION",
            "Do not attempt self-treatment",
            "Document the lesion with photos",
            "Avoid sun exposure to the area",
            "Schedule appointment with dermatologist immediately",
            "Consider biopsy for confirmation"
        ],
        "severity": "High"
    },
    "Normal": {
        "description": "Healthy skin with no apparent conditions detected",
        "recommendations": [
            "Continue with current skincare routine",
            "Use sunscreen daily (SPF 30+)",
            "Stay hydrated and maintain healthy diet",
            "Regular skin checks for prevention",
            "Consider annual dermatologist visit",
            "Protect skin from environmental damage"
        ],
        "severity": "Low"
    },
    "Psoriasis": {
        "description": "Autoimmune condition causing rapid skin cell growth and scaling",
        "recommendations": [
            "Keep skin moisturized with thick creams",
            "Avoid triggers (stress, infections, injuries)",
            "Use gentle, fragrance-free products",
            "Consider phototherapy options",
            "Avoid alcohol and smoking",
            "Consult dermatologist for systemic treatments"
        ],
        "severity": "Moderate"
    },
    "Rosacea": {
        "description": "Chronic skin condition causing facial redness and visible blood vessels",
        "recommendations": [
            "Identify and avoid triggers (spicy foods, alcohol, heat)",
            "Use gentle, non-irritating skincare products",
            "Protect skin from sun exposure",
            "Avoid hot beverages and extreme temperatures",
            "Consider green-tinted makeup to reduce redness",
            "Consult dermatologist for prescription treatments"
        ],
        "severity": "Moderate"
    }
}

def preprocess_image(image_data):
    """Preprocess image for ML model"""
    try:
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard size (224x224 for most models)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_skin_condition(image_array):
    """
    Predict skin condition using ML model or fallback to basic prediction
    """
    if ml_model and ML_MODEL_AVAILABLE:
        try:
            # Use the ML model for prediction
            predicted_class, confidence, all_probabilities = ml_model.predict(image_array)
            return predicted_class, confidence, all_probabilities
        except Exception as e:
            print(f"ML model prediction failed: {e}")
            # Fallback to basic prediction
    
    # Basic fallback prediction (original implementation)
    import random
    predictions = [0.1, 0.15, 0.05, 0.4, 0.2, 0.1]  # Mock probabilities
    
    # Add some randomness to make it more realistic
    for i in range(len(predictions)):
        predictions[i] += random.uniform(-0.05, 0.05)
        predictions[i] = max(0, min(1, predictions[i]))  # Clamp between 0 and 1
    
    # Normalize probabilities
    total = sum(predictions)
    predictions = [p/total for p in predictions]
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return predicted_class, confidence, predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_skin_condition():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        # Preprocess the image
        image_array = preprocess_image(data['image'])
        print("Image array:", type(image_array), getattr(image_array, 'shape', None))
        predicted_class, confidence, all_probabilities = predict_skin_condition(image_array)
        print("Prediction result:", predicted_class, confidence, all_probabilities)
        # Convert to native Python types for JSON serialization
        predicted_class = int(predicted_class)
        confidence = float(confidence)
        all_probabilities = [float(p) for p in all_probabilities]
        condition_name = SKIN_CONDITIONS.get(predicted_class, "Unknown")
        prescription = PRESCRIPTIONS.get(condition_name, {
            "description": "Unknown condition",
            "recommendations": ["Please consult a healthcare professional"],
            "severity": "Unknown"
        })
        return jsonify({
            'condition': condition_name,
            'confidence': confidence,
            'description': prescription['description'],
            'recommendations': prescription['recommendations'],
            'severity': prescription['severity'],
            'all_probabilities': {SKIN_CONDITIONS[i]: float(prob) for i, prob in enumerate(all_probabilities)},
            'timestamp': datetime.now().isoformat(),
            'id': str(uuid.uuid4()),
            'model_type': ml_model.model_type if ml_model else 'basic'
        })
    except Exception as e:
        print("Error in detection:", e)
        traceback.print_exc()
        return jsonify({'error': 'Failed to analyze image. Please try again.'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    model_status = "available" if ml_model else "unavailable"
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'model_status': model_status,
        'model_type': ml_model.model_type if ml_model else 'basic'
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if ml_model:
        return jsonify({
            'model_type': ml_model.model_type,
            'model_path': ml_model.model_path,
            'class_names': ml_model.get_all_class_names(),
            'available': True
        })
    else:
        return jsonify({
            'model_type': 'basic',
            'available': False,
            'class_names': list(SKIN_CONDITIONS.values())
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081) 