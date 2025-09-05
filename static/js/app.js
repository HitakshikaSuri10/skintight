// Global variables
let selectedFile = null;
let imagePreview = null;
let analysisResults = null;
let detectedRegions = [];

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsCard = document.getElementById('resultsCard');
const conditionsCard = document.getElementById('conditionsCard');
const placeholderCard = document.getElementById('placeholderCard');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const regionCount = document.getElementById('regionCount');
const conditionsList = document.getElementById('conditionsList');
const resultCanvas = document.getElementById('resultCanvas');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateConfidenceDisplay();
});

function initializeEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload area drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Confidence slider
    confidenceSlider.addEventListener('input', updateConfidenceDisplay);
    
    // Tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
}

function handleDrop(event) {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    selectedFile = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview = e.target.result;
        displayImagePreview();
    };
    reader.readAsDataURL(file);
    
    // Simulate upload progress
    simulateUploadProgress();
}

function displayImagePreview() {
    uploadContent.innerHTML = `
        <div class="upload-preview">
            <img src="${imagePreview}" alt="Preview" class="preview-image">
            <p class="preview-filename">${selectedFile.name}</p>
            <div class="upload-progress" id="uploadProgress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="uploadProgressFill"></div>
                </div>
                <p class="progress-text">Uploading... <span id="uploadProgressText">0</span>%</p>
            </div>
        </div>
    `;
}

function simulateUploadProgress() {
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressFill = document.getElementById('uploadProgressFill');
    const uploadProgressText = document.getElementById('uploadProgressText');
    
    uploadProgress.style.display = 'block';
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        uploadProgressFill.style.width = progress + '%';
        uploadProgressText.textContent = progress;
        
        if (progress >= 100) {
            clearInterval(interval);
            uploadProgress.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    }, 100);
}

function updateConfidenceDisplay() {
    confidenceValue.textContent = confidenceSlider.value;
}

function scrollToUpload() {
    document.getElementById('analysis-tool').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

function openCamera() {
    alert('Camera functionality would be available in the full implementation');
}

async function startAnalysis() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="lucide lucide-loader-2 spinner"></i> Analyzing...';
    progressContainer.style.display = 'block';
    
    // Simulate analysis steps
    const steps = [
        'Preprocessing image...',
        'Detecting skin regions...',
        'Analyzing texture patterns...',
        'Identifying potential conditions...',
        'Calculating confidence scores...',
        'Generating recommendations...'
    ];
    
    for (let i = 0; i < steps.length; i++) {
        progressText.textContent = steps[i];
        progressFill.style.width = ((i + 1) / steps.length * 100) + '%';
        await new Promise(resolve => setTimeout(resolve, 800));
    }
    
    // Send to backend for analysis
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            analysisResults = await response.json();
            displayResults();
        } else {
            throw new Error('Analysis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed. Please try again.');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="lucide lucide-eye"></i> Start Analysis';
        progressContainer.style.display = 'none';
    }
}

function displayResults() {
    if (!analysisResults) return;
    
    // Hide placeholder, show results
    placeholderCard.style.display = 'none';
    resultsCard.style.display = 'block';
    conditionsCard.style.display = 'block';
    
    // Draw annotated image
    drawAnnotatedImage();
    
    // Display conditions
    displayConditions();
    
    // Update region count
    regionCount.textContent = `${analysisResults.detected_regions.length} region(s) detected`;
}

function drawAnnotatedImage() {
    if (!imagePreview || !analysisResults.detected_regions) return;
    
    const canvas = resultCanvas;
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw detected regions
        analysisResults.detected_regions.forEach(region => {
            const color = region.condition === 'Mild Acne' ? '#10b981' : '#f59e0b';
            
            // Draw rectangle
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(region.x, region.y, region.width, region.height);
            
            // Draw label background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(region.x, region.y - 20, region.width, 20);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.font = '12px Inter';
            ctx.fillText(`${region.condition} (${region.confidence}%)`, region.x + 2, region.y - 5);
        });
    };
    img.src = imagePreview;
}

function displayConditions() {
    if (!analysisResults.conditions || analysisResults.conditions.length === 0) {
        conditionsList.innerHTML = `
            <div class="text-center py-8 text-muted-foreground">
                <i class="lucide lucide-check-circle" style="width: 3rem; height: 3rem; margin: 0 auto 1rem; color: var(--color-success);"></i>
                <p class="text-lg font-medium">No significant conditions detected</p>
                <p class="text-sm">Your skin appears healthy based on the analysis</p>
            </div>
        `;
        return;
    }
    
    const conditionsHtml = analysisResults.conditions.map(condition => `
        <div class="condition-item">
            <div class="condition-header">
                <h3 class="condition-title">${condition.condition}</h3>
                <div class="condition-badges">
                    <span class="severity-badge severity-${condition.severity}">${condition.severity}</span>
                    <span class="confidence-badge">${condition.confidence}% confidence</span>
                </div>
            </div>
            
            <p class="condition-description">${condition.description}</p>
            
            <div class="recommendations">
                <h4 class="recommendations-title">Recommendations:</h4>
                <ul class="recommendations-list">
                    ${condition.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `).join('');
    
    conditionsList.innerHTML = conditionsHtml;
}

// Utility functions
function getSeverityColor(severity) {
    switch (severity) {
        case 'mild': return 'severity-mild';
        case 'moderate': return 'severity-moderate';
        case 'severe': return 'severity-severe';
        default: return 'severity-mild';
    }
}

// Export functions for global access
window.scrollToUpload = scrollToUpload;
window.openCamera = openCamera;
window.startAnalysis = startAnalysis;
