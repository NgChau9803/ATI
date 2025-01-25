from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import base64
from PIL import Image
import io
import cv2
import torch as torch
import numpy as np

app = Flask(__name__)
CORS(app)

def get_onnx_providers():
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']

# Load the ONNX model
session = ort.InferenceSession(
    'model/advanced_mnist_model.onnx', 
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

    image_array = np.array(image, dtype=np.float32)

    # Apply adaptive thresholding
    image_array = cv2.adaptiveThreshold(
        image_array.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    ).astype(np.float32)
    
    # Normalize using MNIST parameters (same as training)
    image_array = (image_array / 255.0 - 0.1307) / 0.3081
    
    # Reshape for model input and ensure float32
    input_tensor = image_array.reshape(1, 1, 28, 28).astype(np.float32)
    
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
        
        logits = outputs[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Get prediction
        prediction = int(np.argmax(probabilities))
        probabilities_list = probabilities[0].tolist()
        
        return jsonify({
            'digit': prediction,
            'outputs': outputs[0][0].tolist(),
            'probabilities': probabilities_list
        })
    
    # Error handling
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)