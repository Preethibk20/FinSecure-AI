// Enhanced JavaScript for Deep Learning Spam Detector

class SpamDetectorApp {
    constructor() {
        this.currentAnalysis = null;
        this.modelStatus = {};
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.checkModelStatus();
        this.initializeCharts();
    }

    setupEventListeners() {
        // Main analyze button
        document.getElementById('analyze-btn').addEventListener('click', () => {
            this.analyzeText();
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearResults();
        });

        // Demo button
        document.getElementById('demo-button').addEventListener('click', () => {
            this.loadDemoText();
        });

        // Compare models button
        document.getElementById('compare-models-btn').addEventListener('click', () => {
            this.compareAllModels();
        });

        // Compare all button (in comparison tab)
        document.getElementById('compare-all-btn').addEventListener('click', () => {
            this.compareAllModelsFromTab();
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Result tab switching
        document.querySelectorAll('.result-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchResultTab(e.target.dataset.resultTab);
            });
        });

        // Model selection change
        document.querySelectorAll('input[name="model"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.updateModelSelection();
            });
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/api/models/status');
            const status = await response.json();
            this.modelStatus = status;
            this.updateModelStatusUI();
        } catch (error) {
            console.error('Error checking model status:', error);
            this.showError('Failed to check model availability');
        }
    }

    updateModelStatusUI() {
        const statusElements = {
            'traditional_ml': document.getElementById('traditional-status'),
            'deep_learning': document.getElementById('deep_learning-status'),
            'ensemble': document.getElementById('ensemble-status')
        };

        Object.keys(statusElements).forEach(modelType => {
            const element = statusElements[modelType];
            const isAvailable = this.modelStatus[modelType]?.available;
            
            if (isAvailable) {
                element.innerHTML = '<i class="fas fa-check-circle status-available"></i> Available';
                element.className = 'model-status status-available';
            } else {
                element.innerHTML = '<i class="fas fa-times-circle status-unavailable"></i> Unavailable';
                element.className = 'model-status status-unavailable';
                
                // Disable radio button if model is not available
                const radio = document.getElementById(modelType);
                if (radio) {
                    radio.disabled = true;
                    radio.parentElement.style.opacity = '0.5';
                }
            }
        });

        // Auto-select first available model
        this.selectFirstAvailableModel();
    }

    selectFirstAvailableModel() {
        const modelOrder = ['ensemble', 'deep_learning', 'traditional_ml'];
        const radioIds = {
            'ensemble': 'ensemble',
            'deep_learning': 'deep_learning',
            'traditional_ml': 'traditional'
        };
        
        for (const modelType of modelOrder) {
            if (this.modelStatus[modelType]?.available) {
                const radioId = radioIds[modelType];
                const radio = document.getElementById(radioId);
                if (radio && !radio.disabled) {
                    radio.checked = true;
                    break;
                }
            }
        }
    }

    updateModelSelection() {
        const selectedModel = document.querySelector('input[name="model"]:checked').value;
        console.log('Selected model:', selectedModel);
    }

    async analyzeText() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to analyze');
            return;
        }

        const selectedModel = document.querySelector('input[name="model"]:checked').value;
        
        this.showLoader('Analyzing text with AI...');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model_type: selectedModel
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.currentAnalysis = result;
            this.displayResults(result);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Failed to analyze text. Please try again.');
        } finally {
            this.hideLoader();
        }
    }

    async compareAllModels() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to compare');
            return;
        }

        this.showLoader('Comparing across all models...');
        
        try {
            const response = await fetch(`/api/models/compare?text=${encodeURIComponent(text)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayModelComparison(result);
            
        } catch (error) {
            console.error('Comparison error:', error);
            this.showError('Failed to compare models. Please try again.');
        } finally {
            this.hideLoader();
        }
    }

    async compareAllModelsFromTab() {
        const textInput = document.getElementById('comparison-text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to compare');
            return;
        }

        this.showLoader('Comparing across all models...');
        
        try {
            const response = await fetch(`/api/models/compare?text=${encodeURIComponent(text)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayComparisonResults(result);
            
        } catch (error) {
            console.error('Comparison error:', error);
            this.showError('Failed to compare models. Please try again.');
        } finally {
            this.hideLoader();
        }
    }

    displayResults(result) {
        // Update prediction box
        this.updatePredictionBox(result);
        
        // Show ensemble breakdown if applicable
        if (result.ensemble_results) {
            this.showEnsembleBreakdown(result.ensemble_results);
        } else {
            document.getElementById('ensemble-results').style.display = 'none';
        }
        
        // Update charts
        this.updateCharts(result);
        
        // Update URL analysis
        this.updateURLAnalysis(result.urls || []);
        
        // Show results container
        document.getElementById('results-container').style.display = 'block';
        
        // Switch to spam analysis tab
        this.switchResultTab('spam-analysis');
    }

    updatePredictionBox(result) {
        const predictionBox = document.getElementById('prediction-box');
        const predictionIcon = predictionBox.querySelector('.prediction-icon');
        const predictionTitle = predictionBox.querySelector('.prediction-title');
        const predictionDescription = predictionBox.querySelector('.prediction-description');
        const confidenceFill = predictionBox.querySelector('.confidence-fill');
        const confidenceValue = predictionBox.querySelector('.confidence-value');
        const modelUsedText = document.getElementById('model-used-text');
        
        const isSpam = result.prediction === 'Spam';
        
        // Update icon and styling
        predictionIcon.innerHTML = isSpam ? 
            '<i class="fas fa-exclamation-triangle"></i>' : 
            '<i class="fas fa-check-circle"></i>';
        
        predictionBox.className = `prediction-box ${isSpam ? 'spam' : 'ham'}`;
        
        // Update text
        predictionTitle.textContent = result.prediction;
        predictionDescription.textContent = isSpam ? 
            'This message appears to be spam. Be cautious with any links or requests.' :
            'This message appears to be legitimate.';
        
        // Update confidence bar
        confidenceFill.style.width = `${result.confidence}%`;
        confidenceValue.textContent = `${result.confidence.toFixed(1)}%`;
        
        // Update model used
        modelUsedText.textContent = result.model_used;
    }

    showEnsembleBreakdown(ensembleResults) {
        const ensembleContainer = document.getElementById('ensemble-results');
        const breakdownContainer = document.getElementById('ensemble-breakdown');
        
        let html = '';
        
        if (ensembleResults.traditional_ml) {
            html += this.createModelResultCard('Traditional ML', ensembleResults.traditional_ml, ensembleResults.weights.traditional_ml);
        }
        
        if (ensembleResults.deep_learning) {
            html += this.createModelResultCard('Deep Learning', ensembleResults.deep_learning, ensembleResults.weights.deep_learning);
        }
        
        breakdownContainer.innerHTML = html;
        ensembleContainer.style.display = 'block';
    }

    createModelResultCard(modelName, result, weight) {
        return `
            <div class="model-result">
                <h5>${modelName} (Weight: ${(weight * 100).toFixed(0)}%)</h5>
                <div class="prediction ${result.prediction === 'Spam' ? 'spam' : 'ham'}">
                    ${result.prediction}
                </div>
                <div class="confidence">
                    Confidence: ${result.confidence.toFixed(1)}%
                </div>
                <div class="confidence">
                    Spam: ${result.spam_probability.toFixed(1)}% | 
                    Ham: ${result.not_spam_probability.toFixed(1)}%
                </div>
            </div>
        `;
    }

    displayModelComparison(result) {
        // This could show a modal or update a section with comparison results
        console.log('Model comparison:', result);
        
        // For now, let's show an alert with the results
        let message = 'Model Comparison Results:\n\n';
        
        Object.keys(result.model_predictions).forEach(modelType => {
            const pred = result.model_predictions[modelType];
            message += `${modelType.toUpperCase()}:\n`;
            message += `  Prediction: ${pred.prediction}\n`;
            message += `  Confidence: ${pred.confidence.toFixed(1)}%\n\n`;
        });
        
        alert(message);
    }

    displayComparisonResults(result) {
        const comparisonContainer = document.getElementById('all-models-comparison');
        const resultsContainer = document.getElementById('comparison-results');
        
        let html = '';
        
        Object.keys(result.model_predictions).forEach(modelType => {
            const pred = result.model_predictions[modelType];
            html += `
                <div class="model-result">
                    <h5>${this.formatModelName(modelType)}</h5>
                    <div class="prediction ${pred.prediction === 'Spam' ? 'spam' : 'ham'}">
                        ${pred.prediction}
                    </div>
                    <div class="confidence">
                        Confidence: ${pred.confidence.toFixed(1)}%
                    </div>
                    <div class="confidence">
                        Spam: ${pred.spam_probability.toFixed(1)}% | 
                        Ham: ${pred.not_spam_probability.toFixed(1)}%
                    </div>
                    <div class="model-status">
                        Model: ${pred.model_used}
                    </div>
                </div>
            `;
        });
        
        comparisonContainer.innerHTML = html;
        resultsContainer.style.display = 'block';
    }

    formatModelName(modelType) {
        const names = {
            'traditional_ml': 'Traditional ML',
            'deep_learning': 'Deep Learning',
            'ensemble': 'Ensemble'
        };
        return names[modelType] || modelType;
    }

    updateCharts(result) {
        this.updateGaugeChart(result.spam_probability);
        this.updatePieChart(result.spam_probability, result.not_spam_probability);
    }

    updateURLAnalysis(urls) {
        const urlCardsContainer = document.getElementById('url-cards-container');
        const noUrlsMessage = document.getElementById('no-urls-message');
        
        if (urls.length === 0) {
            noUrlsMessage.style.display = 'block';
            urlCardsContainer.innerHTML = '';
            return;
        }
        
        noUrlsMessage.style.display = 'none';
        
        let html = '';
        urls.forEach((url, index) => {
            html += this.createURLCard(url, index);
        });
        
        urlCardsContainer.innerHTML = html;
    }

    createURLCard(url, index) {
        const trustClass = this.getTrustClass(url.trust_score);
        
        return `
            <div class="url-card ${trustClass}">
                <div class="url-header">
                    <h4>${url.domain}</h4>
                    <div class="trust-score">
                        <span class="score">${url.trust_score}/100</span>
                        <span class="classification">${url.classification}</span>
                    </div>
                </div>
                <div class="url-details">
                    <p class="url-link">${url.url}</p>
                    ${url.risk_factors.length > 0 ? `
                        <div class="risk-factors">
                            <h5>Risk Factors:</h5>
                            <ul>
                                ${url.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${url.security_features.length > 0 ? `
                        <div class="security-features">
                            <h5>Security Features:</h5>
                            <ul>
                                ${url.security_features.map(feature => `<li>${feature}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    getTrustClass(score) {
        if (score >= 80) return 'high-trust';
        if (score >= 60) return 'moderate-trust';
        if (score >= 40) return 'low-trust';
        return 'suspicious';
    }

    async loadDemoText() {
        try {
            const response = await fetch('/api/demo-text');
            const data = await response.json();
            document.getElementById('text-input').value = data.text;
        } catch (error) {
            console.error('Error loading demo text:', error);
        }
    }

    clearResults() {
        document.getElementById('text-input').value = '';
        document.getElementById('results-container').style.display = 'none';
        document.getElementById('comparison-results').style.display = 'none';
        document.getElementById('comparison-text-input').value = '';
        this.currentAnalysis = null;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
    }

    switchResultTab(tabName) {
        // Update result tab buttons
        document.querySelectorAll('.result-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-result-tab="${tabName}"]`).classList.add('active');
        
        // Update result tab content
        document.querySelectorAll('.result-tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
    }

    showLoader(message = 'Processing...') {
        document.getElementById('status-text').textContent = message;
        document.getElementById('loader').style.display = 'block';
        
        // Animate progress bar
        const progressFill = document.querySelector('.progress-fill');
        progressFill.style.width = '0%';
        
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = `${progress}%`;
        }, 200);
        
        this.progressInterval = interval;
    }

    hideLoader() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        const progressFill = document.querySelector('.progress-fill');
        progressFill.style.width = '100%';
        
        setTimeout(() => {
            document.getElementById('loader').style.display = 'none';
        }, 300);
    }

    showError(message) {
        alert(`Error: ${message}`);
    }

    initializeCharts() {
        // Initialize empty charts
        this.initGaugeChart();
        this.initPieChart();
    }

    initGaugeChart() {
        const ctx = document.getElementById('spam-gauge').getContext('2d');
        
        this.gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#ff6b6b', '#e9ecef'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    initPieChart() {
        const ctx = document.getElementById('classification-pie').getContext('2d');
        
        this.pieChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Spam', 'Not Spam'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#ff6b6b', '#51cf66'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    updateGaugeChart(spamProbability) {
        const nonSpamProbability = 100 - spamProbability;
        
        this.gaugeChart.data.datasets[0].data = [spamProbability, nonSpamProbability];
        this.gaugeChart.update();
        
        // Update center text
        const centerText = document.querySelector('.gauge-center-text');
        centerText.innerHTML = `<div class="gauge-value">${spamProbability.toFixed(1)}%</div>`;
    }

    updatePieChart(spamProbability, notSpamProbability) {
        this.pieChart.data.datasets[0].data = [spamProbability, notSpamProbability];
        this.pieChart.update();
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SpamDetectorApp();
});