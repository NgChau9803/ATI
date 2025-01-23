document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const predictionDisplay = document.getElementById('prediction');
    const probabilityChartContainer = document.getElementById('probability-chart');
    let probabilityChart = null;

    // Canvas setup
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

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

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionDisplay.textContent = '-';
        probabilityChartContainer.classList.add('hidden');
        if (probabilityChart) {
            probabilityChart.destroy();
        }
    }

    function preprocessCanvas() {
        // Create a new canvas for preprocessing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Fill with white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Draw the original canvas onto the smaller canvas with high-quality scaling
        tempCtx.imageSmoothingEnabled = true;
        tempCtx.imageSmoothingQuality = 'high';
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
        
        return tempCanvas.toDataURL('image/png');
    }

    function predict() {
        // Preprocess the canvas and get image data
        const imageData = preprocessCanvas();

        // Log the image data for debugging
        console.log('Image Data URL Length:', imageData.length);
        console.log('Image Data (first 100 chars):', imageData.substring(0, 100) + '...');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => {
            // Log the raw response for debugging
            console.log('Response status:', response.status);
            console.log('Response headers:', Object.fromEntries(response.headers.entries()));
            return response.json();
        })
        .then(data => {
            // Extensive logging of the response data
            console.log('Full prediction data:', data);
            
            // Validate the response data
            if (!data || data.digit === undefined) {
                throw new Error('Invalid prediction response');
            }

            // Update prediction display
            predictionDisplay.textContent = data.digit;

            // Check if raw_outputs exists and is an array
            if (data.raw_outputs && Array.isArray(data.raw_outputs) && data.raw_outputs.length > 0) {
                updateProbabilityChart(data.raw_outputs[0]);
            } else {
                console.warn('No raw outputs found in the prediction data');
                probabilityChartContainer.classList.add('hidden');
            }
        })
        .catch(error => {
            // Comprehensive error handling
            console.error('Prediction Error:', error);
            
            // Detailed error message display
            predictionDisplay.textContent = 'Error';
            probabilityChartContainer.classList.add('hidden');
            
            // Optional: Show a more detailed error message
            alert(`Prediction failed: ${error.message}`);
        });
    }

    function updateProbabilityChart(probabilities) {
        // Validate probabilities
        if (!probabilities || !Array.isArray(probabilities) || probabilities.length !== 10) {
            console.warn('Invalid probabilities:', probabilities);
            probabilityChartContainer.classList.add('hidden');
            return;
        }

        probabilityChartContainer.classList.remove('hidden');
        const ctx = document.getElementById('probabilityCanvas').getContext('2d');
        
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 10}, (_, i) => i),
                datasets: [{
                    label: 'Digit Probabilities',
                    data: probabilities,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Probability'
                        }
                    }
                }
            }
        });
    }

    predictBtn.addEventListener('click', predict);
    clearBtn.addEventListener('click', clearCanvas);
});