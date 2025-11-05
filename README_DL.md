# Advanced Spam Text Detector with Deep Learning

An enhanced spam detection system that combines traditional machine learning with deep learning models for superior accuracy. This project implements multiple AI approaches including LSTM, CNN, and ensemble methods.

## 🚀 Features

### AI Models
- **Deep Learning Models**: LSTM, CNN-LSTM, and Ensemble neural networks
- **Traditional ML**: Naive Bayes with TF-IDF (baseline)
- **Ensemble Method**: Weighted combination of all models for optimal accuracy
- **Model Comparison**: Side-by-side analysis of different approaches

### Advanced Analysis
- **Text Classification**: Multi-model spam detection
- **URL Safety Analysis**: Domain trust scoring and risk assessment
- **Real-time Processing**: Fast inference with multiple model options
- **Interactive UI**: Modern web interface with model selection

### Technical Features
- **Bidirectional LSTM**: Context-aware text understanding
- **Convolutional Layers**: Pattern detection in text
- **Word Embeddings**: Semantic text representation
- **Ensemble Voting**: Weighted model combination
- **REST API**: Full API access to all models

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.15+
- FastAPI
- scikit-learn
- NLTK
- pandas, numpy, matplotlib

## 🛠️ Installation

### Quick Setup
```bash
# Clone or download the project
cd spam-text-detector

# Run automated setup
python setup.py

# Or manual installation:
pip install -r requirements.txt
python train_models.py
python main_dl.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Train models (requires mail_data.csv)
python train_models.py --model both

# Start the application
python main_dl.py
```

## 📊 Dataset

Download the SMS Spam Collection Dataset:
- **Source**: [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **File**: Save as `mail_data.csv` in the project root
- **Format**: CSV with columns: `Category`, `Message`

## 🧠 Model Architecture

### Deep Learning Models

#### 1. LSTM Model
```python
- Embedding Layer (10k vocab, 128 dim)
- Bidirectional LSTM (64 units) + Dropout
- Bidirectional LSTM (32 units) + Dropout  
- Dense (64) + BatchNorm + Dropout
- Dense (32) + Dropout
- Output (1, sigmoid)
```

#### 2. CNN-LSTM Hybrid
```python
- Embedding Layer (10k vocab, 128 dim)
- Conv1D (128 filters, kernel=5) + MaxPool
- Conv1D (64 filters, kernel=5) + MaxPool
- LSTM (64 units) + Dropout
- Dense (64) + BatchNorm + Dropout
- Output (1, sigmoid)
```

#### 3. Ensemble Model
```python
- LSTM Branch: Bidirectional LSTM + GlobalMaxPool
- CNN Branch: Conv1D + GlobalMaxPool
- Concatenate branches
- Dense layers with dropout
- Output (1, sigmoid)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Epochs**: 30 (with early stopping)

## 🚀 Usage

### Web Application
```bash
# Start the enhanced application
python main_dl.py

# Or start the original version
python main.py

# Open browser to http://localhost:8000
```

### API Usage

#### Analyze Text
```python
import requests

response = requests.post('http://localhost:8000/api/analyze', 
    json={
        'text': 'Congratulations! You won $1000!',
        'model_type': 'ensemble'  # or 'deep_learning', 'traditional'
    }
)
result = response.json()
```

#### Compare Models
```python
response = requests.get('http://localhost:8000/api/models/compare', 
    params={'text': 'Your text here'}
)
comparison = response.json()
```

#### Check Model Status
```python
response = requests.get('http://localhost:8000/api/models/status')
status = response.json()
```

### Command Line Training
```bash
# Train all models
python train_models.py --model both --epochs 30

# Train only deep learning models
python train_models.py --model deep_learning --epochs 50

# Train only traditional model
python train_models.py --model traditional
```

### Python API
```python
from deep_learning_model import SpamDetectorDL

# Load trained model
detector = SpamDetectorDL.load_model('spam_detector_ensemble.h5', 'tokenizer_ensemble.pkl')

# Predict
result = detector.predict("Free money! Click here now!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

## 📈 Model Performance

### Expected Results
- **Traditional ML (Naive Bayes)**: ~96% accuracy
- **LSTM Model**: ~97-98% accuracy  
- **CNN-LSTM Hybrid**: ~97-98% accuracy
- **Ensemble Model**: ~98-99% accuracy

### Performance Metrics
- **Precision**: Minimize false positives
- **Recall**: Catch all spam messages
- **F1-Score**: Balanced performance
- **Inference Speed**: <100ms per prediction

## 🏗️ Project Structure

```
spam-text-detector/
├── deep_learning_model.py      # DL model implementation
├── main_dl.py                  # Enhanced FastAPI app
├── train_models.py             # Training script
├── setup.py                    # Automated setup
├── requirements.txt            # Dependencies
├── mail_data.csv              # Dataset (download required)
│
├── templates/
│   ├── index.html             # Original interface
│   └── index_dl.html          # Enhanced interface
│
├── static/
│   ├── css/styles.css         # Styling
│   ├── js/main.js            # Original JS
│   └── js/main_dl.js         # Enhanced JS
│
├── models/ (generated)
│   ├── spam_detector_lstm.h5
│   ├── spam_detector_ensemble.h5
│   ├── tokenizer_lstm.pkl
│   └── text_classification.pkl
│
└── outputs/ (generated)
    ├── training_history_dl.png
    ├── confusion_matrix_dl.png
    └── model_comparison.png
```

## 🔧 Configuration

### Model Parameters
```python
# In deep_learning_model.py
SpamDetectorDL(
    max_features=10000,    # Vocabulary size
    max_length=100,        # Sequence length
    embedding_dim=128      # Embedding dimensions
)
```

### Training Parameters
```python
# Training configuration
epochs=30
batch_size=32
validation_split=0.2
early_stopping_patience=10
```

### Ensemble Weights
```python
# In main_dl.py - analyze_with_ensemble()
dl_weight = 0.7           # Deep learning model weight
traditional_weight = 0.3   # Traditional ML weight
```

## 🎯 Model Selection Guide

### When to Use Each Model

#### Ensemble (Recommended)
- **Best overall accuracy**
- **Robust predictions**
- **Transparent decision breakdown**
- Use when: Maximum accuracy is needed

#### Deep Learning
- **High accuracy**
- **Context understanding**
- **Pattern recognition**
- Use when: Complex text patterns expected

#### Traditional ML
- **Fast inference**
- **Low resource usage**
- **Interpretable results**
- Use when: Speed and simplicity are priorities

## 🔍 Advanced Features

### Text Preprocessing
- Lowercasing and cleaning
- Stop word removal
- Stemming with Porter Stemmer
- Sequence padding and truncation

### URL Analysis
- Domain reputation scoring
- SSL certificate validation
- Phishing pattern detection
- URL shortener identification

### Visualization
- Training history plots
- Confusion matrices
- Model comparison charts
- Real-time confidence meters

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Retrain models if corrupted
python train_models.py --model both
```

#### Memory Issues
```python
# Reduce batch size in training
batch_size=16  # Instead of 32
```

#### NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

#### TensorFlow GPU Issues
```bash
# Install CPU version if GPU issues
pip install tensorflow-cpu==2.15.0
```

## 📚 API Reference

### Endpoints

#### POST /api/analyze
Analyze text with selected model
- **Body**: `{"text": "string", "model_type": "ensemble|deep_learning|traditional"}`
- **Response**: Prediction results with confidence scores

#### GET /api/models/compare
Compare all models on given text
- **Params**: `text` (string)
- **Response**: Side-by-side model predictions

#### GET /api/models/status
Check model availability
- **Response**: Status of all models

#### GET /api/demo-text
Get sample text for testing
- **Response**: Demo text with spam indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SMS Spam Collection Dataset from UCI ML Repository
- TensorFlow and Keras teams
- FastAPI framework
- scikit-learn library
- NLTK project

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Create an issue on GitHub
4. Check model training logs

---

**Happy Spam Detecting! 🛡️**