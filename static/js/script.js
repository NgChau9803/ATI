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

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        const pos = getCanvasCoordinates(e);
        [lastX, lastY] = [pos.x, pos.y];
    }

    function draw(e) {
        if (!isDrawing) return;
        const pos = getCanvasCoordinates(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        [lastX, lastY] = [pos.x, pos.y];
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


    // Softmax function to convert raw outputs to probabilities
    function softmax(outputs) {
        const outputArray = Array.isArray(outputs) ? outputs : 
                        typeof outputs === 'object' ? Object.values(outputs) : 
                        [outputs];
    
        const exp = outputArray.map(x => Math.exp(x));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sumExp);
    }      

    function updateProbabilityChart(rawOutputs) {
        // Convert raw outputs to probabilities
        const probabilities = softmax(rawOutputs);

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
                    backgroundColor: probabilities.map((prob, index) => 
                        index === parseInt(predictionDisplay.textContent) 
                        ? 'rgba(59, 130, 246, 0.8)' // Highlight predicted digit
                        : 'rgba(59, 130, 246, 0.4)'
                    ),
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
                            text: 'Probability',
                            font: {
                                weight: 'bold',
                                size: 20
                            }
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Digit',
                            font: {
                                weight: 'bold',
                                size: 20
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return (context.parsed.y * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    function predict() {
        // Preprocess the canvas and get image data
        const imageData = preprocessCanvas();

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => {
            return response.json();
        })
        .then(data => {
            // Validate the response data
            if (!data || data.digit === undefined) {
                throw new Error('Invalid prediction response');
            }

            // Update prediction display
            predictionDisplay.textContent = data.digit;

            // Check if outputs exists and is an array
            if (data.outputs && Array.isArray(data.outputs) && data.outputs.length > 0) {
                updateProbabilityChart(data.outputs);
            } else {
                console.warn('No outputs found in the prediction data');
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

    predictBtn.addEventListener('click', predict);
    clearBtn.addEventListener('click', clearCanvas);
});