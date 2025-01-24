document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const predictionDisplay = document.getElementById('prediction');
    const probabilityChartContainer = document.getElementById('probability-chart');
    let probabilityChart = null;
    
    // History variables
    let predictionHistory = [];
    const historyContainer = document.getElementById('history-container');

    // Brush settings
    let currentColor = 'black';
    let currentSize = 15;

    // Initialize canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = currentSize;
    ctx.lineCap = 'round';
    ctx.strokeStyle = currentColor;

    // Brush color selection
    document.querySelectorAll('.brush-color').forEach(button => {
        button.addEventListener('click', () => {
            currentColor = button.dataset.color;
            ctx.strokeStyle = currentColor;
            
            // Update active state
            document.querySelectorAll('.brush-color').forEach(btn => 
                btn.classList.remove('ring-2', 'ring-offset-2', 'ring-blue-500'));
            button.classList.add('ring-2', 'ring-offset-2', 'ring-blue-500');
        });
    });

    // Brush size control
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeValue = document.getElementById('brushSizeValue');
    brushSizeInput.addEventListener('input', (e) => {
        currentSize = parseInt(e.target.value);
        ctx.lineWidth = currentSize;
        brushSizeValue.textContent = currentSize;
    });

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
        
        // Reset brush settings
        ctx.strokeStyle = currentColor;
        ctx.lineWidth = currentSize;
    }

    function preprocessCanvas() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        tempCtx.imageSmoothingEnabled = true;
        tempCtx.imageSmoothingQuality = 'high';
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
        
        return tempCanvas.toDataURL('image/png');
    }

    function softmax(outputs) {
        const outputArray = Array.isArray(outputs) ? outputs : 
                        typeof outputs === 'object' ? Object.values(outputs) : 
                        [outputs];
    
        const exp = outputArray.map(x => Math.exp(x));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sumExp);
    }

    function updateProbabilityChart(rawOutputs) {
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
                        ? 'rgba(59, 130, 246, 0.8)'
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

    function updatePredictionHistory(imageData, prediction, confidence) {
        const historyItem = {
            image: imageData,
            prediction: prediction,
            confidence: (confidence * 100).toFixed(1) + '%',
            timestamp: new Date().toLocaleTimeString()
        };

        predictionHistory.unshift(historyItem);
        if (predictionHistory.length > 15) {
            predictionHistory.pop();
        }

        historyContainer.innerHTML = '';

        predictionHistory.forEach(item => {
            const historyElement = document.createElement('div');
            historyElement.className = 'history-item bg-gray-50 p-3 rounded-lg cursor-pointer';
            historyElement.innerHTML = `
                <div class="flex items-center space-x-3">
                    <img src="${item.image}" class="w-12 h-12 object-contain bg-white p-1 rounded border" alt="Drawn digit">
                    <div>
                        <div class="font-semibold text-blue-600">Prediction: ${item.prediction}</div>
                        <div class="text-sm text-gray-500">${item.confidence}</div>
                        <div class="text-xs text-gray-400">${item.timestamp}</div>
                    </div>
                </div>
            `;
            historyContainer.appendChild(historyElement);
        });
    }

    function predict() {
        const imageData = preprocessCanvas();

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (!data || data.digit === undefined) {
                throw new Error('Invalid prediction response');
            }

            predictionDisplay.textContent = data.digit;

            if (data.outputs && Array.isArray(data.outputs) && data.outputs.length > 0) {
                updateProbabilityChart(data.outputs);
            } else {
                console.warn('No outputs found in the prediction data');
                probabilityChartContainer.classList.add('hidden');
            }

            // Update prediction history
            const confidence = Math.max(...data.probabilities);
            updatePredictionHistory(imageData, data.digit, confidence);
        })
        .catch(error => {
            console.error('Prediction Error:', error);
            predictionDisplay.textContent = 'Error';
            probabilityChartContainer.classList.add('hidden');
            alert(`Prediction failed: ${error.message}`);
        });
    }

    predictBtn.addEventListener('click', predict);
    clearBtn.addEventListener('click', clearCanvas);
});