# ASL Recognition System - Deployment Guide

## Overview
This is a complete American Sign Language (ASL) alphabet recognition system built with MediaPipe and LSTM neural networks.

## Performance
- **Accuracy**: 98.3%
- **F1-Score**: 0.984
- **Prediction Time**: 3.5ms
- **Real-time Capable**: Yes (25+ FPS)

## Files Included
- `asl_recognition_system_v1.h5` - Trained Keras model
- `asl_recognition_system_v1_preprocessor.pkl` - Data preprocessor
- `asl_recognition_system_v1_metadata.json` - Complete system metadata
- `asl_recognition_system_v1_training_history.pkl` - Training history
- `asl_recognition_system_v1_evaluation_results.pkl` - Evaluation results

## Requirements
```
tensorflow>=2.15.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## Usage Example
```python
import tensorflow as tf
import pickle
import numpy as np

# Load model and preprocessor
model = tf.keras.models.load_model('asl_recognition_system_v1.h5')
with open('asl_recognition_system_v1_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Process input sequence (30 frames × 42 features)
# sequence = your_landmark_sequence
# normalized_seq = preprocessor._normalize_sequences(np.array([sequence]), fit=False)
# prediction = model.predict(normalized_seq)
# label, confidence = preprocessor.decode_prediction(prediction[0])
```

## Classes Supported
A, B, C, E, G, H, I, J, K, L, S, T, U, V, W, Y, Z

## Notes
- Input sequences must be 30 frames long
- Each frame contains 42 hand landmark features (21 landmarks × 2 coordinates)
- Landmarks should be normalized relative to wrist position
- Confidence threshold of 0.7 recommended for stable predictions
