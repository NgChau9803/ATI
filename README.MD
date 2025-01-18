# Handwritten Digit Recognition Web Application

## Project Overview

A web-based machine learning application that recognizes handwritten digits using a neural network trained on the MNIST dataset.

## Features

- Real-time digit recognition
- Web interface for drawing digits
- Machine learning model with 97.74% accuracy
- ONNX model export for cross-platform compatibility

## Technology Stack

- **Backend**:
  - Python
  - Flask
  - PyTorch
  - ONNX Runtime
- **Frontend**:
  - HTML5 Canvas
  - JavaScript
- **Machine Learning**:
  - MNIST Dataset
  - Convolutional Neural Network

## Prerequisites

- Python 3.8+
- pip
- Virtual Environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NgChau9803/ATI
cd ATI
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main.py to download the dataset and train the model

```bash
python main.py
```

### 5. Run the app.py to start the web application

```bash
python app.py
```

## Usage

1. Open the web application in your browser.
2. Draw a digit on the canvas.
3. Click on the "Predict" button to get the predicted digit.
