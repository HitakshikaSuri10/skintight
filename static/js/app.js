// Global variables
let currentImage = null;
let isAnalyzing = false;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const progressFill = document.getElementById('progressFill');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

function setupEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        currentImage = e.target.result;
        displayImagePreview(currentImage);
    };
    reader.readAsDataURL(file);
}

function displayImagePreview(imageData) {
    previewImage.src = imageData;
    uploadArea.style.display = 'none';
    imagePreview.style.display = 'block';
}

function resetUpload() {
    currentImage = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    uploadArea.classList.remove('dragover');
}

// Analysis functions
async function analyzeImage() {
    if (!currentImage || isAnalyzing) return;
    
    isAnalyzing = true;
    showLoading();
    
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Simulate loading time for better UX
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Failed to analyze image. Please try again.');
    } finally {
        isAnalyzing = false;
        hideLoading();
    }
}

function showLoading() {
    uploadSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Animate progress bar
    animateProgressBar();
}

function hideLoading() {
    loadingSection.style.display = 'none';
}

function animateProgressBar() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
        }
        progressFill.style.width = progress + '%';
    }, 200);
}

function displayResults(result) {
    // Update condition information
    document.getElementById('conditionName').textContent = result.condition;
    document.getElementById('confidenceBadge').textContent = result.confidence + '%';
    document.getElementById('conditionDescription').textContent = result.description;
    
    // Update severity badge
    const severityBadge = document.getElementById('severityBadge');
    severityBadge.textContent = result.severity;
    severityBadge.className = 'severity-badge ' + result.severity.toLowerCase();
    
    // Update recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    recommendationsList.innerHTML = '';
    result.recommendations.forEach(recommendation => {
        const li = document.createElement('li');
        li.textContent = recommendation;
        recommendationsList.appendChild(li);
    });
    
    // Update probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    Object.entries(result.all_probabilities).forEach(([condition, probability]) => {
        const probabilityItem = document.createElement('div');
        probabilityItem.className = 'probability-item';
        
        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = condition;
        
        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        
        const fill = document.createElement('div');
        fill.className = 'probability-fill';
        fill.style.width = '0%';
        
        const value = document.createElement('div');
        value.className = 'probability-value';
        value.textContent = probability + '%';
        
        bar.appendChild(fill);
        probabilityItem.appendChild(label);
        probabilityItem.appendChild(bar);
        probabilityItem.appendChild(value);
        probabilityBars.appendChild(probabilityItem);
        
        // Animate the probability bar
        setTimeout(() => {
            fill.style.width = probability + '%';
        }, 100);
    });
    
    // Show results
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function resetAnalysis() {
    resetUpload();
    uploadSection.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Error handling
function showError(message) {
    const errorModal = document.getElementById('errorModal');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorModal.style.display = 'flex';
}

function closeModal() {
    const errorModal = document.getElementById('errorModal');
    errorModal.style.display = 'none';
}

// Utility functions
function showPrivacyPolicy() {
    alert('Privacy Policy: Your uploaded images are processed securely and are not stored permanently. We use industry-standard encryption to protect your data.');
}

function showTerms() {
    alert('Terms of Service: This application is for educational and informational purposes only. It should not replace professional medical advice. Always consult with a qualified healthcare provider.');
}

// Close modal when clicking outside
window.addEventListener('click', function(event) {
    const modal = document.getElementById('errorModal');
    if (event.target === modal) {
        closeModal();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Add some visual feedback for better UX
function addVisualFeedback(element, className) {
    element.classList.add(className);
    setTimeout(() => {
        element.classList.remove(className);
    }, 200);
}

// Enhance upload area with visual feedback
uploadArea.addEventListener('click', function() {
    addVisualFeedback(this, 'clicked');
});

// Add loading state to analyze button
analyzeBtn.addEventListener('click', function() {
    if (!isAnalyzing) {
        addVisualFeedback(this, 'clicked');
    }
}); 