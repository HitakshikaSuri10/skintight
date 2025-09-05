# AI Skin Detection App

A modern web application for AI-powered skin condition analysis using machine learning and computer vision.

## Features

- **AI-Powered Analysis**: Upload skin photos for instant AI analysis
- **Modern UI**: Beautiful, responsive design with drag-and-drop functionality
- **Real-time Results**: Get detailed analysis with confidence scores and recommendations
- **Multiple Condition Detection**: Detects acne, rosacea, and general skin health issues
- **Professional Interface**: Medical disclaimers and privacy-focused design

## Technology Stack

- **Backend**: Python Flask
- **ML Framework**: TensorFlow
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, Pillow
- **Styling**: Custom CSS with modern design system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HitakshikaSuri10/skin-detect-app.git
cd skin-detect-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python3 app.py
```

4. Open your browser and navigate to `http://localhost:5001`

## Usage

1. **Upload Image**: Drag and drop or click to upload a clear photo of your skin
2. **Configure Analysis**: Choose analysis type (General, Acne, or Rosacea) and confidence threshold
3. **Start Analysis**: Click "Start Analysis" to begin AI processing
4. **View Results**: See annotated images with detected conditions and personalized recommendations

## Model Information

The application uses a pre-trained TensorFlow model (`skin_model_finetuned.h5`) that automatically downloads on first run. The model is optimized for skin condition detection and analysis.

## Medical Disclaimer

This application is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified dermatologist or healthcare provider for proper medical evaluation.

## Privacy

Your uploaded images are processed securely and are not stored on our servers after analysis completion. All data is encrypted in transit and processed in compliance with healthcare privacy standards.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.
