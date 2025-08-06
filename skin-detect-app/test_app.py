Skip to content
Navigation Menu
HitakshikaSuri10
skintight

Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Files
Go to file
t
skin-detect-app
app.py
ml_model.py
requirements.txt
run.py
static
templates
.gitignore
.python-version
README.md
app.py
ml_model.py
render.yaml
requirements.txt
run.py
test_app.py
train_skin_model.py
train_skin_model_old.py
skintight
/test_app.py
Hitakshika SuriHitakshika Suri
Hitakshika Suri
and
Hitakshika Suri
Clean repository with latest updates including About Me section
a31c241
 · 
15 hours ago

Code

Blame
172 lines (144 loc) · 5.73 KB
#!/usr/bin/env python3
"""
Test script for SkinSight application

This script tests the basic functionality of the skin detection application
including the ML model integration and API endpoints.
"""

import requests
import json
import numpy as np
from PIL import Image
import io
import base64
import time

def create_test_image():
    """Create a test image for testing"""
    # Create a simple test image (224x224 RGB)
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model Status: {data.get('model_status')}")
            print(f"   Model Type: {data.get('model_type')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/api/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model Type: {data.get('model_type')}")
            print(f"   Available: {data.get('available')}")
            print(f"   Classes: {data.get('class_names')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_detection_endpoint(base_url):
    """Test the skin detection endpoint"""
    print("\n🔍 Testing detection endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare request data
        data = {
            'image': test_image
        }
        
        # Make request
        response = requests.post(
            f"{base_url}/api/detect",
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detection successful")
            print(f"   Condition: {result.get('condition')}")
            print(f"   Confidence: {result.get('confidence')}%")
            print(f"   Severity: {result.get('severity')}")
            print(f"   Model Type: {result.get('model_type')}")
            
            # Print probabilities
            print("   Probabilities:")
            for condition, prob in result.get('all_probabilities', {}).items():
                print(f"     {condition}: {prob}%")
            
            return True
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

def test_ml_model():
    """Test the ML model directly"""
    print("\n🔍 Testing ML model directly...")
    try:
        from ml_model import create_mock_model
        
        # Create model
        model = create_mock_model()
        
        # Create test image
        test_image = np.random.random((224, 224, 3))
        
        # Make prediction
        predicted_class, confidence, probabilities = model.predict(test_image)
        
        print(f"✅ ML model test passed")
        print(f"   Predicted Class: {model.get_class_name(predicted_class)}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Model Type: {model.model_type}")
        
        return True
    except Exception as e:
        print(f"❌ ML model test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 SkinSight - Test Suite")
    print("=" * 50)
    
    # Test configuration
    base_url = "http://localhost:5000"
    
    # Test ML model directly
    ml_test_passed = test_ml_model()
    
    # Wait a moment for server to start if needed
    print("\n⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test API endpoints
    health_passed = test_health_endpoint(base_url)
    model_info_passed = test_model_info_endpoint(base_url)
    detection_passed = test_detection_endpoint(base_url)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ML Model Test: {'✅ PASSED' if ml_test_passed else '❌ FAILED'}")
    print(f"   Health Endpoint: {'✅ PASSED' if health_passed else '❌ FAILED'}")
    print(f"   Model Info Endpoint: {'✅ PASSED' if model_info_passed else '❌ FAILED'}")
    print(f"   Detection Endpoint: {'✅ PASSED' if detection_passed else '❌ FAILED'}")
    
    all_passed = ml_test_passed and health_passed and model_info_passed and detection_passed
    
    if all_passed:
        print("\n🎉 All tests passed! The application is working correctly.")
        print("🌐 You can now open your browser and go to http://localhost:5000")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        print("💡 Make sure the Flask server is running: python run.py")

if __name__ == "__main__":
    main() 
skintight/test_app.py at main · HitakshikaSuri10/skintight 
