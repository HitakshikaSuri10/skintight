"""
Machine Learning Model Integration for Skin Condition Detection

This module provides functions to load and use a trained machine learning model
for skin condition classification. The current implementation includes a mock model,
but you can easily integrate your own trained model.

Author: SkinSight Team
Date: 2024
"""

import numpy as np
import cv2
from PIL import Image
import os
import pickle
import json

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using mock model.")

# Try to import scikit-learn (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Using mock model.")

class SkinConditionModel:
    """
    Main class for skin condition detection model.
    Supports multiple model types and provides a unified interface.
    """
    
    def __init__(self, model_path=None, model_type='mock'):
        """
        Initialize the skin condition detection model.
        
        Args:
            model_path (str): Path to the trained model file
            model_type (str): Type of model ('tensorflow', 'sklearn', 'mock')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.class_names = [
            "Acne", "Eczema", "Melanoma", "Normal", "Psoriasis", "Rosacea"
        ]
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on type."""
        if self.model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
            self._load_tensorflow_model()
        elif self.model_type == 'sklearn' and SKLEARN_AVAILABLE:
            self._load_sklearn_model()
        else:
            print("Using mock model for demonstration purposes.")
            self.model = 'mock'
    
    def _load_tensorflow_model(self):
        """Load a TensorFlow/Keras model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"TensorFlow model loaded from {self.model_path}")
            else:
                print("Model file not found. Using mock model.")
                self.model = 'mock'
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            self.model = 'mock'
    
    def _load_sklearn_model(self):
        """Load a scikit-learn model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load the model and scaler
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"Scikit-learn model loaded from {self.model_path}")
            else:
                print("Model file not found. Using mock model.")
                self.model = 'mock'
        except Exception as e:
            print(f"Error loading scikit-learn model: {e}")
            self.model = 'mock'
    
    def preprocess_image(self, image_array):
        """
        Preprocess image for model input.
        
        Args:
            image_array (numpy.ndarray): Input image array (224x224x3)
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model
        """
        # Ensure image is in the correct format
        if len(image_array.shape) == 3:
            # Convert to grayscale if needed (for some models)
            # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            # image_array = np.expand_dims(image_array, axis=-1)
            pass
        
        # Normalize pixel values
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def extract_features(self, image_array):
        """
        Extract features from image (for traditional ML models).
        
        Args:
            image_array (numpy.ndarray): Input image array
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Convert to grayscale for feature extraction
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Extract basic features
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten()[:50])  # First 50 histogram bins
        
        # Statistical features
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.var(gray))
        features.append(np.median(gray))
        
        # Texture features (simplified)
        # You can add more sophisticated texture features here
        features.append(np.max(gray))
        features.append(np.min(gray))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, image_array):
        """
        Make prediction on the input image.
        
        Args:
            image_array (numpy.ndarray): Input image array (224x224x3)
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        if self.model == 'mock':
            return self._mock_predict()
        
        try:
            if self.model_type == 'tensorflow':
                return self._tensorflow_predict(image_array)
            elif self.model_type == 'sklearn':
                return self._sklearn_predict(image_array)
            else:
                return self._mock_predict()
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_predict()
    
    def _tensorflow_predict(self, image_array):
        """Make prediction using TensorFlow model."""
        # Preprocess image
        processed_image = self.preprocess_image(image_array)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        probabilities = predictions[0]
        
        # Get results
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def _sklearn_predict(self, image_array):
        """Make prediction using scikit-learn model."""
        # Extract features
        features = self.extract_features(image_array)
        
        # Scale features if scaler is available
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Make prediction
        predicted_class = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def _mock_predict(self):
        """Mock prediction for demonstration purposes."""
        # Simulate realistic predictions with some randomness
        import random
        
        # Base probabilities (can be adjusted)
        base_probabilities = [0.1, 0.15, 0.05, 0.4, 0.2, 0.1]
        
        # Add randomness
        probabilities = []
        for prob in base_probabilities:
            # Add some random variation
            variation = random.uniform(-0.05, 0.05)
            new_prob = max(0, min(1, prob + variation))
            probabilities.append(new_prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def get_class_name(self, class_index):
        """Get the name of a class by its index."""
        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return "Unknown"
    
    def get_all_class_names(self):
        """Get all class names."""
        return self.class_names.copy()

# Example usage and model training functions
def create_mock_model():
    """Create a mock model for demonstration."""
    return SkinConditionModel(model_type='mock')

def train_simple_model(data_path, save_path):
    """
    Train a simple scikit-learn model (example function).
    
    Args:
        data_path (str): Path to training data
        save_path (str): Path to save the trained model
    """
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available for training.")
        return
    
    # This is a placeholder for actual training code
    # In a real implementation, you would:
    # 1. Load your training data
    # 2. Extract features from images
    # 3. Train the model
    # 4. Save the model
    
    print("Training functionality not implemented in this demo.")
    print("Please implement your own training pipeline based on your dataset.")

def load_pretrained_model(model_path, model_type='auto'):
    """
    Load a pretrained model.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model ('tensorflow', 'sklearn', 'auto')
        
    Returns:
        SkinConditionModel: Loaded model instance
    """
    if model_type == 'auto':
        # Auto-detect model type based on file extension
        if model_path.endswith('.h5') or model_path.endswith('.pb'):
            model_type = 'tensorflow'
        elif model_path.endswith('.pkl') or model_path.endswith('.pickle'):
            model_type = 'sklearn'
        else:
            model_type = 'mock'
    
    return SkinConditionModel(model_path=model_path, model_type=model_type)

# Example usage
if __name__ == "__main__":
    # Create a mock model for testing
    model = create_mock_model()
    
    # Test with a dummy image
    dummy_image = np.random.random((224, 224, 3))
    
    # Make prediction
    predicted_class, confidence, probabilities = model.predict(dummy_image)
    
    print(f"Predicted class: {model.get_class_name(predicted_class)}")
    print(f"Confidence: {confidence:.2%}")
    print("All probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {model.get_class_name(i)}: {prob:.2%}") 
