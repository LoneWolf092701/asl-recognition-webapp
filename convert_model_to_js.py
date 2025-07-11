# convert_model_to_js.py - FIXED VERSION
import tensorflow as tf
import tensorflowjs as tfjs
import json
import pickle
import numpy as np
import os

def convert_model_to_web():
    """Convert the trained ASL model to web-compatible format"""
    
    print("🔄 Converting ASL model for web deployment...")
    
    # Load your trained model (update path if needed)
    model_path = "models/asl_recognition_system_v1/asl_recognition_system_v1.h5"
    
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = "best_asl_model.h5"
        
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please check the path.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create web model directory
    web_model_path = "model"
    os.makedirs(web_model_path, exist_ok=True)
    
    # Convert to TensorFlow.js
    try:
        tfjs.converters.save_keras_model(model, web_model_path)
        print(f"✅ Model converted to TensorFlow.js at: {web_model_path}/")
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return
    
    # Load preprocessor for normalization parameters
    preprocessor_path = "models/asl_recognition_system_v1/asl_recognition_system_v1_preprocessor.pkl"
    
    if not os.path.exists(preprocessor_path):
        print("❌ Preprocessor file not found. Creating dummy normalization params...")
        # Create dummy params if preprocessor not found
        normalization_params = {
            'mean': [0.0] * 42,  # 42 features
            'std': [1.0] * 42,
            'class_names': ['A', 'B', 'C', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
        }
    else:
        try:
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            
            # Extract normalization parameters
            normalization_params = {
                'mean': preprocessor.feature_stats['mean'].tolist(),
                'std': preprocessor.feature_stats['std'].tolist(),
                'class_names': preprocessor.label_encoder.classes_.tolist()
            }
            print("✅ Preprocessor loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading preprocessor: {e}")
            print("Creating default normalization params...")
            normalization_params = {
                'mean': [0.0] * 42,
                'std': [1.0] * 42,
                'class_names': ['A', 'B', 'C', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
            }
    
    # Save normalization parameters for web
    with open("normalization_params.json", "w") as f:
        json.dump(normalization_params, f, indent=2)
    
    print(f"✅ Normalization parameters saved to: normalization_params.json")
    
    # Print summary
    print("\n🎉 CONVERSION COMPLETED!")
    print("📁 Files created:")
    print(f"  • model/model.json")
    print(f"  • model/weights.bin")  
    print(f"  • normalization_params.json")
    print(f"\n📊 Model info:")
    print(f"  • Classes: {len(normalization_params['class_names'])}")
    print(f"  • Features: {len(normalization_params['mean'])}")
    print(f"  • Classes: {normalization_params['class_names']}")
    
    return True

if __name__ == "__main__":
    # Install required package first
    print("Installing tensorflowjs...")
    os.system("pip install tensorflowjs")
    
    # Run conversion
    convert_model_to_web()