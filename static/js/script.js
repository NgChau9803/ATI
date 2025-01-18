const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const predictButton = document.getElementById('predictButton');
const predictedDigitSpan = document.getElementById('predictedDigit');
const confidenceSpan = document.getElementById('confidence');

// Drawing setup
ctx.lineWidth = 20;  // Increased line width
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';
ctx.fillStyle = 'white';

// Fill canvas with white background
ctx.fillRect(0, 0, canvas.width, canvas.height);

let lastX = 0;
let lastY = 0;
let isDrawing = false;

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictedDigitSpan.textContent = '-';
    confidenceSpan.textContent = '-';
}

async function predictDigit() {
    // Create a new canvas to process the image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with white background
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Draw the original canvas onto the smaller canvas
    // Use LANCZOS resampling for better quality
    tempCtx.imageSmoothingEnabled = true;
    tempCtx.imageSmoothingQuality = 'high';
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    // Debug: Save the image locally
    const debugImage = tempCanvas.toDataURL('image/png');
    console.log('Debug Image Data URL Length:', debugImage.length);
    console.log('Debug Image:', debugImage.substring(0, 100) + '...');
    
    // Optional: Visually confirm the small canvas
    const link = document.createElement('a');
    link.href = debugImage;
    link.download = 'debug_input.png';
    link.click();
    
    // Convert to base64
    const imageBase64 = tempCanvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageBase64 })
        });
        
        const result = await response.json();
        
        // Log raw result for debugging
        console.log('Prediction Result:', result);
        
        predictedDigitSpan.textContent = result.digit;
        confidenceSpan.textContent = (result.confidence * 100).toFixed(2);
        
        // Optional: Log raw outputs
        if (result.raw_outputs) {
            console.log('Raw Outputs:', result.raw_outputs);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        console.error('Full error:', JSON.stringify(error, null, 2));
    }
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predictDigit);