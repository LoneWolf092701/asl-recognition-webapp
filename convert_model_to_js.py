# convert_model_to_js.py
import tensorflow as tf
import tensorflowjs as tfjs
import json
import pickle
import numpy as np

def convert_model_to_tensorflowjs():
    """Convert the trained ASL model to TensorFlow.js format"""
    
    # Load your trained model
    model_path = "models/asl_recognition_system_v1/asl_recognition_system_v1.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow.js
    tfjs_model_path = "web_app/model"
    tfjs.converters.save_keras_model(model, tfjs_model_path)
    
    print(f"✅ Model converted to TensorFlow.js at: {tfjs_model_path}")
    
    # Load preprocessor for normalization parameters
    with open("models/asl_recognition_system_v1/asl_recognition_system_v1_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    
    # Export normalization parameters
    normalization_params = {
        'mean': preprocessor.feature_stats['mean'].tolist(),
        'std': preprocessor.feature_stats['std'].tolist(),
        'class_names': preprocessor.label_encoder.classes_.tolist()
    }
    
    # Save as JSON for JavaScript
    with open("web_app/normalization_params.json", "w") as f:
        json.dump(normalization_params, f, indent=2)
    
    print(f"✅ Normalization parameters saved")
    
    return tfjs_model_path

if __name__ == "__main__":
    # Install required package first: pip install tensorflowjs
    convert_model_to_tensorflowjs()