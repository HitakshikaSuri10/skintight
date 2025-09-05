import os
import requests
import base64
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_URL = "https://drive.google.com/uc?export=download&id=13ABkJk8mvR7QUCNzyLawe6E57NHcggEe"
MODEL_PATH = "skin_model_finetuned.h5"

# Global variable to store the loaded model
model = None

def download_model_if_missing():
    """
    Download the model file from Google Drive if it does not exist locally.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Downloading from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded model to {MODEL_PATH}")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")
    else:
        print(f"Model file {MODEL_PATH} already exists.")

def load_model():
    """
    Load the TensorFlow model for skin analysis.
    """
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction.
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def analyze_skin_condition(image_path):
    """
    Analyze the skin condition using the loaded model.
    """
    model = load_model()
    if model is None:
        return None
    
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None
    
    try:
        # Make prediction
        prediction = model.predict(processed_img)
        
        # Mock analysis results for demonstration
        # In a real implementation, you would process the model output
        conditions = [
            {
                'condition': 'Mild Acne',
                'confidence': min(85 + np.random.randint(-10, 10), 100),
                'severity': 'mild',
                'description': 'Small comedones and minor inflammatory lesions detected in the T-zone area.',
                'recommendations': [
                    'Use a gentle salicylic acid cleanser twice daily',
                    'Apply a non-comedogenic moisturizer',
                    'Consider over-the-counter benzoyl peroxide treatment'
                ]
            },
            {
                'condition': 'Sun Damage',
                'confidence': min(72 + np.random.randint(-10, 10), 100),
                'severity': 'moderate',
                'description': 'Signs of photoaging including hyperpigmentation and texture irregularities.',
                'recommendations': [
                    'Apply broad-spectrum SPF 30+ sunscreen daily',
                    'Consider vitamin C serum for antioxidant protection',
                    'Consult dermatologist for professional treatment options'
                ]
            }
        ]
        
        # Filter results based on confidence threshold
        filtered_conditions = [cond for cond in conditions if cond['confidence'] >= 70]
        
        return {
            'conditions': filtered_conditions,
            'detected_regions': [
                {'x': 120, 'y': 80, 'width': 40, 'height': 30, 'condition': 'Mild Acne', 'confidence': 85},
                {'x': 200, 'y': 150, 'width': 60, 'height': 45, 'condition': 'Sun Damage', 'confidence': 72}
            ]
        }
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

@app.route('/')
def index():
    """Main page with the integrated design."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for skin analysis."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            analysis_result = analyze_skin_condition(filepath)
            
            if analysis_result:
                # Convert image to base64 for display
                with open(filepath, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                analysis_result['image_base64'] = img_base64
                analysis_result['filename'] = filename
                
                return jsonify(analysis_result)
            else:
                return jsonify({'error': 'Analysis failed'}), 500
    
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Download the model if needed before starting the app
    download_model_if_missing()
    app.run(debug=True, host='0.0.0.0', port=5001)