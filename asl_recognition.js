// asl_recognition.js - FIXED VERSION with correct paths
class ASLRecognitionSystem {
    constructor() {
        this.model = null;
        this.normalizationParams = null;
        this.hands = null;
        this.camera = null;
        this.isRunning = false;
        
        // Sequence buffer for temporal analysis
        this.sequenceBuffer = [];
        this.sequenceLength = 30;
        
        // Performance tracking
        this.frameCount = 0;
        this.predictionCount = 0;
        this.lastTime = performance.now();
        
        // UI elements
        this.initializeElements();
        
        // Initialize the system
        this.init();
    }
    
    initializeElements() {
        this.webcamElement = document.getElementById('webcam');
        this.canvasElement = document.getElementById('output-canvas');
        this.canvasCtx = this.canvasElement.getContext('2d');
        
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.resetBtn = document.getElementById('reset-btn');
        
        this.predictedLetterElement = document.getElementById('predicted-letter');
        this.confidenceFillElement = document.getElementById('confidence-fill');
        this.confidenceTextElement = document.getElementById('confidence-text');
        
        this.fpsValueElement = document.getElementById('fps-value');
        this.predictionsCountElement = document.getElementById('predictions-count');
        this.accuracyValueElement = document.getElementById('accuracy-value');
        
        this.historyContainer = document.getElementById('history-container');
        this.loadingOverlay = document.getElementById('loading-overlay');
        this.errorModal = document.getElementById('error-modal');
        
        // Event listeners
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.resetBtn.addEventListener('click', () => this.reset());
    }
    
    async init() {
        try {
            await this.loadModel();
            await this.loadNormalizationParams();
            await this.initializeMediaPipe();
            this.hideLoading();
        } catch (error) {
            this.showError('Failed to initialize ASL Recognition System: ' + error.message);
            console.error('Initialization error:', error);
        }
    }
    
    async loadModel() {
        try {
            this.updateProgress(20, 'Loading AI model...');
            
            // FIXED: Correct path to your model files
            console.log('Loading model from: ./model/model.json');
            this.model = await tf.loadLayersModel('./model/model.json');
            
            console.log('Model loaded successfully');
            console.log('Model input shape:', this.model.inputShape);
            this.updateProgress(40, 'Model loaded successfully');
        } catch (error) {
            console.error('Model loading error:', error);
            throw new Error('Failed to load model. Please ensure model files are present in ./model/ directory: ' + error.message);
        }
    }
    
    async loadNormalizationParams() {
        try {
            this.updateProgress(60, 'Loading normalization parameters...');
            
            // FIXED: Correct path to normalization parameters
            console.log('Loading normalization params from: ./normalization_params.json');
            const response = await fetch('./normalization_params.json');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            this.normalizationParams = await response.json();
            console.log('Normalization parameters loaded');
            console.log('Classes:', this.normalizationParams.class_names);
            console.log('Features:', this.normalizationParams.mean.length);
            this.updateProgress(80, 'Configuration loaded');
        } catch (error) {
            console.error('Normalization params loading error:', error);
            throw new Error('Failed to load normalization parameters: ' + error.message);
        }
    }
    
    async initializeMediaPipe() {
        try {
            this.updateProgress(90, 'Initializing hand tracking...');
            
            // Check if MediaPipe is available
            if (typeof Hands === 'undefined') {
                throw new Error('MediaPipe Hands not loaded. Check if MediaPipe scripts are included.');
            }
            
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });
            
            this.hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.5
            });
            
            this.hands.onResults(this.onResults.bind(this));
            console.log('MediaPipe initialized');
            this.updateProgress(100, 'Ready!');
        } catch (error) {
            console.error('MediaPipe initialization error:', error);
            throw new Error('Failed to initialize MediaPipe: ' + error.message);
        }
    }
    
    async startCamera() {
        try {
            const constraints = {
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            this.webcamElement.srcObject = stream;
            this.webcamElement.addEventListener('loadeddata', () => {
                this.canvasElement.width = this.webcamElement.videoWidth;
                this.canvasElement.height = this.webcamElement.videoHeight;
                this.startRecognition();
            });
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.isRunning = true;
            
        } catch (error) {
            console.error('Camera access error:', error);
            this.showError('Failed to access camera. Please ensure camera permissions are granted and camera is not in use by another application.');
        }
    }
    
    stopCamera() {
        this.isRunning = false;
        
        if (this.webcamElement.srcObject) {
            this.webcamElement.srcObject.getTracks().forEach(track => track.stop());
            this.webcamElement.srcObject = null;
        }
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Clear canvas
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
    
    reset() {
        this.sequenceBuffer = [];
        this.frameCount = 0;
        this.predictionCount = 0;
        this.lastTime = performance.now();
        
        this.predictedLetterElement.textContent = '?';
        this.confidenceFillElement.style.width = '0%';
        this.confidenceTextElement.textContent = '0%';
        
        this.fpsValueElement.textContent = '0';
        this.predictionsCountElement.textContent = '0';
        this.accuracyValueElement.textContent = '0%';
        
        this.historyContainer.innerHTML = '<p class="no-history">No predictions yet. Start the camera to begin recognition.</p>';
        
        // Clear active alphabet highlighting
        document.querySelectorAll('.alphabet-item').forEach(item => {
            item.classList.remove('active');
        });
    }
    
    startRecognition() {
        if (!this.isRunning) return;
        
        // Check if Camera class is available
        if (typeof Camera === 'undefined') {
            console.error('MediaPipe Camera class not available');
            this.showError('MediaPipe Camera utilities not loaded. Please check internet connection.');
            return;
        }
        
        this.camera = new Camera(this.webcamElement, {
            onFrame: async () => {
                if (this.isRunning && this.hands) {
                    await this.hands.send({ image: this.webcamElement });
                }
            }
        });
        
        this.camera.start();
    }
    
    onResults(results) {
        if (!this.isRunning) return;
        
        // Clear canvas
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        
        // Draw camera feed
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvasElement.width, this.canvasElement.height);
        
        // Process hand landmarks
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            
            // Draw hand landmarks
            if (typeof drawConnectors !== 'undefined' && typeof HAND_CONNECTIONS !== 'undefined') {
                drawConnectors(this.canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
                drawLandmarks(this.canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
            } else {
                // Fallback: draw simple landmarks
                this.drawSimpleLandmarks(landmarks);
            }
            
            // Extract features
            const features = this.extractFeatures(landmarks);
            const normalizedFeatures = this.normalizeFeatures(features);
            
            // Add to sequence buffer
            this.sequenceBuffer.push(normalizedFeatures);
            if (this.sequenceBuffer.length > this.sequenceLength) {
                this.sequenceBuffer.shift();
            }
            
            // Make prediction if buffer is full
            if (this.sequenceBuffer.length === this.sequenceLength) {
                this.makePrediction();
            }
        }
        
        this.canvasCtx.restore();
        
        // Update FPS
        this.updateFPS();
    }
    
    drawSimpleLandmarks(landmarks) {
        // Fallback method if MediaPipe drawing utils not available
        this.canvasCtx.fillStyle = '#FF0000';
        
        for (const landmark of landmarks) {
            this.canvasCtx.beginPath();
            this.canvasCtx.arc(
                landmark.x * this.canvasElement.width,
                landmark.y * this.canvasElement.height,
                3,
                0,
                2 * Math.PI
            );
            this.canvasCtx.fill();
        }
    }
    
    extractFeatures(landmarks) {
        const features = [];
        
        // Extract x, y coordinates for all 21 landmarks
        for (let i = 0; i < landmarks.length; i++) {
            features.push(landmarks[i].x);
            features.push(landmarks[i].y);
        }
        
        return features;
    }
    
    normalizeFeatures(features) {
        if (!this.normalizationParams) {
            console.warn('Normalization parameters not loaded, using raw features');
            return features;
        }
        
        const normalized = [];
        const { mean, std } = this.normalizationParams;
        
        // Ensure we have the right number of features
        if (features.length !== mean.length) {
            console.warn(`Feature length mismatch: got ${features.length}, expected ${mean.length}`);
            return features;
        }
        
        for (let i = 0; i < features.length; i++) {
            normalized[i] = (features[i] - mean[i]) / (std[i] + 1e-8); // Add small epsilon to avoid division by zero
        }
        
        return normalized;
    }
    
    async makePrediction() {
        try {
            // Prepare input tensor
            const inputShape = [1, this.sequenceLength, 42]; // [batch, time, features]
            const inputTensor = tf.tensor3d([this.sequenceBuffer], inputShape);
            
            // Make prediction
            const prediction = this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Get predicted class
            const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[maxProbIndex];
            const predictedLetter = this.normalizationParams.class_names[maxProbIndex];
            
            // Update UI only if confidence is high enough
            if (confidence > 0.6) { // Lowered threshold for better responsiveness
                this.updatePrediction(predictedLetter, confidence);
                this.addToHistory(predictedLetter, confidence);
                this.highlightAlphabet(predictedLetter);
                this.predictionCount++;
            }
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();
            
        } catch (error) {
            console.error('Prediction error:', error);
        }
    }
    
    updatePrediction(letter, confidence) {
        this.predictedLetterElement.textContent = letter;
        this.confidenceFillElement.style.width = (confidence * 100) + '%';
        this.confidenceTextElement.textContent = Math.round(confidence * 100) + '%';
        
        // Update prediction count
        this.predictionsCountElement.textContent = this.predictionCount;
    }
    
    addToHistory(letter, confidence) {
        // Remove "no history" message if present
        const noHistoryMsg = this.historyContainer.querySelector('.no-history');
        if (noHistoryMsg) {
            noHistoryMsg.remove();
        }
        
        // Create history item
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.textContent = `${letter} (${Math.round(confidence * 100)}%)`;
        
        // Add to container
        this.historyContainer.appendChild(historyItem);
        
        // Keep only last 20 predictions
        const historyItems = this.historyContainer.querySelectorAll('.history-item');
        if (historyItems.length > 20) {
            historyItems[0].remove();
        }
    }
    
    highlightAlphabet(letter) {
        // Remove previous highlights
        document.querySelectorAll('.alphabet-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Highlight current letter
        const currentItem = document.querySelector(`[data-letter="${letter}"]`);
        if (currentItem) {
            currentItem.classList.add('active');
        }
    }
    
    updateFPS() {
        this.frameCount++;
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastTime;
        
        if (deltaTime >= 1000) {
            const fps = (this.frameCount * 1000) / deltaTime;
            this.fpsValueElement.textContent = Math.round(fps);
            
            // Calculate accuracy (simplified)
            const accuracy = this.frameCount > 0 ? (this.predictionCount / this.frameCount * 100) : 0;
            this.accuracyValueElement.textContent = Math.round(accuracy) + '%';
            
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }
    
    updateProgress(percent, message) {
        const progressFill = document.getElementById('progress-fill');
        const loadingText = document.querySelector('.loading-content p');
        
        if (progressFill) {
            progressFill.style.width = percent + '%';
        }
        
        if (loadingText) {
            loadingText.textContent = message;
        }
    }
    
    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }
    
    showError(message) {
        this.hideLoading();
        document.getElementById('error-message').textContent = message;
        this.errorModal.classList.remove('hidden');
    }
}

// Hand connections for MediaPipe (fallback if not loaded from CDN)
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17] // Palm
];

// Utility functions
function closeModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

// Initialize the system when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing ASL Recognition System...');
    new ASLRecognitionSystem();
});

// Service worker registration for offline support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}