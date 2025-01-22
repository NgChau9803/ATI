from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import base64
from PIL import Image
import io
import os
import cv2
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Ensure debug directory exists
os.makedirs('debug', exist_ok=True)

def get_onnx_providers():
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']

# Load the ONNX model
session = ort.InferenceSession(
    'model/mnist_model.onnx', 
    providers=get_onnx_providers()
)

def preprocess_image(image_base64):
    # Remove data URL prefix
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    
    try:
        # Open and convert to grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
    except Exception as e:
        print(f"Image Opening Error: {e}")
        with open('debug/problematic_base64.txt', 'w') as f:
            f.write(image_base64)
        raise
    
    # Debug: print image details
    print("Image Mode:", image.mode)
    print("Image Size:", image.size)
    
    # Save original input for debugging
    image.save('debug/original_input.png')
    
    # Check if image is completely black
    image_array = np.array(image)
    print("Original Image Array Stats:")
    print("Min:", image_array.min())
    print("Max:", image_array.max())
    print("Mean:", image_array.mean())
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array and invert (white digit on black background)
    image_array = 255 - np.array(image, dtype=np.float32)
    
    # Normalize using MNIST parameters (same as training)
    image_array = (image_array / 255.0 - 0.1307) / 0.3081
    
    # Reshape for model input and ensure float32
    input_tensor = image_array.reshape(1, 1, 28, 28).astype(np.float32)
    
    # Debug print
    print("Input Tensor Shape:", input_tensor.shape)
    print("Input Tensor Dtype:", input_tensor.dtype)
    print("Input Tensor Min/Max:", input_tensor.min(), input_tensor.max())
    
    return input_tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive base64 encoded image
        image_base64 = request.json['image']
        
        # Preprocess
        input_tensor = preprocess_image(image_base64)
        
        # Run inference
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)
        
        # Get prediction
        prediction = int(np.argmax(outputs[0]))
        
        # Detailed logging
        print("\n--- Prediction Details ---")
        print("Raw Outputs:", outputs[0])
        print("Full Output Array:", outputs[0][0])
        print(f"Prediction: {prediction}")
        
        # Detailed probability visualization
        plt.figure(figsize=(12, 6))
        probabilities = outputs[0][0]
        plt.bar(range(10), probabilities)
        plt.title('Digit Prediction Probabilities')
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.xticks(range(10))
        for i, prob in enumerate(probabilities):
            plt.text(i, prob, f'{prob:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('debug/detailed_probabilities.png')
        plt.close()
        
        return jsonify({
            'digit': prediction,
            'raw_outputs': outputs[0].tolist()
        })
    
    # Error handling
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)