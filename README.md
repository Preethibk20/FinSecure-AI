# 🛡️ Advanced Spam Text Detector

A comprehensive spam detection system that combines traditional machine learning with deep learning models for superior accuracy. This project implements multiple AI approaches including LSTM, CNN, and ensemble methods with an interactive web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### 🤖 Multiple AI Models
- **Deep Learning Models**: LSTM, CNN-LSTM, and Ensemble neural networks
- **Traditional ML**: Naive Bayes with TF-IDF (baseline)
- **Ensemble Method**: Weighted combination of all models for optimal accuracy
- **Real-time Model Switching**: Compare different algorithms instantly

### 🔍 Advanced Analysis
- **Text Classification**: Multi-model spam detection with confidence scores
- **URL Safety Analysis**: Domain trust scoring and phishing detection
- **Interactive Comparison**: Side-by-side model performance analysis
- **Real-time Processing**: Fast inference with multiple model options

### 🎨 Modern Web Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Model Selection**: Easy switching between AI algorithms
- **Visual Analytics**: Charts and confidence meters
- **Dark/Light Theme**: Modern UI with smooth animations

## 📸 Screenshots

### Main Interface
```
┌─────────────────────────────────────────────────────────┐
│                🛡️ Advanced Spam Text Detector           │
├─────────────────────────────────────────────────────────┤
│  🤖 Select AI Model                                    │
│  ○ Ensemble (Best accuracy)     ✅ Available           │
│  ○ Deep Learning (LSTM/CNN)     ✅ Available           │
│  ○ Traditional ML (Naive Bayes) ✅ Available           │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Enter text to analyze...                            │ │
│  │                                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│  [🔍 Analyze] [🗑️ Clear] [⚖️ Compare All Models]        │
└─────────────────────────────────────────────────────────┘
```

### Results Display
```
┌─────────────────────────────────────────────────────────┐
│  🚨 SPAM DETECTED                                       │
│  Confidence: ████████████████░░░░ 89.3%                │
│  Model: Deep Learning (LSTM/CNN)                       │
│                                                         │
│  📊 Ensemble Breakdown:                                 │
│  Traditional ML (30%): Not Spam (62.1%)               │
│  Deep Learning (70%):  Spam (89.3%)                   │
│  Final Decision:       Spam (78.5%)                   │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/spam-text-detector.git
cd spam-text-detector

# Run automated setup
python setup.py

# Start the application
python main_dl.py
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Train models (requires dataset)
python train_models.py --model both

# Start enhanced application
python main_dl.py

# Or start original version
python main.py
```

## 📊 Dataset

**Required**: SMS Spam Collection Dataset
- **Source**: [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **File**: Save as `mail_data.csv` in project root
- **Format**: CSV with columns: `Category`, `Message`
- **Size**: ~5,572 SMS messages (ham/spam labeled)

## 🧠 Model Architecture

### Deep Learning Models

#### 1. LSTM Model
```python
Sequential([
    Embedding(10000, 128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
    Bidirectional(LSTM(32, dropout=0.3)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

#### 2. CNN-LSTM Hybrid
```python
Sequential([
    Embedding(10000, 128, input_length=100),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(5),
    LSTM(64, dropout=0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

#### 3. Ensemble Architecture
- **LSTM Branch**: Bidirectional LSTM + GlobalMaxPooling
- **CNN Branch**: Conv1D + GlobalMaxPooling  
- **Fusion**: Concatenate + Dense layers
- **Output**: Sigmoid activation for binary classification

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Data Split**: 70% train, 15% validation, 15% test

## 🚀 Usage

### Web Application
```bash
# Enhanced version with deep learning
python main_dl.py

# Original version (traditional ML only)
python main.py

# Open browser
http://localhost:8000
```

### API Usage

#### Analyze Text
```python
import requests

# Single model analysis
response = requests.post('http://localhost:8000/api/analyze', 
    json={
        'text': 'URGENT: Your account has been compromised!',
        'model_type': 'ensemble'  # or 'deep_learning', 'traditional'
    }
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

#### Compare All Models
```python
# Compare all models on same text
response = requests.get('http://localhost:8000/api/models/compare', 
    params={'text': 'Free money! Click here now!'}
)
comparison = response.json()

for model, result in comparison['model_predictions'].items():
    print(f"{model}: {result['prediction']} ({result['confidence']:.1f}%)")
```

#### Check Model Status
```python
response = requests.get('http://localhost:8000/api/models/status')
status = response.json()
print("Available models:", list(status.keys()))
```

### Command Line Training
```bash
# Train all models
python train_models.py --model both --epochs 30

# Train specific model
python train_models.py --model deep_learning --epochs 50
python train_models.py --model traditional

# Get help
python train_models.py --help
```

### Python API
```python
from deep_learning_model import SpamDetectorDL

# Load trained model
detector = SpamDetectorDL.load_model(
    'spam_detector_ensemble.h5', 
    'tokenizer_ensemble.pkl'
)

# Make predictions
result = detector.predict("Congratulations! You won $1000!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Spam Probability: {result['spam_probability']:.1f}%")
```

## 📈 Performance Metrics

### Expected Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Traditional ML | ~96% | ~95% | ~94% | ~94% |
| LSTM | ~97-98% | ~96% | ~95% | ~95% |
| CNN-LSTM | ~97-98% | ~96% | ~95% | ~95% |
| **Ensemble** | **~98-99%** | **~97%** | **~96%** | **~96%** |

### Inference Speed
- **Traditional ML**: <10ms per prediction
- **Deep Learning**: <100ms per prediction
- **Ensemble**: <150ms per prediction

## 🏗️ Project Structure

```
spam-text-detector/
├── 📄 README.md                   # This file
├── 📄 requirements.txt            # Python dependencies
├── 📄 setup.py                    # Automated setup script
├── 📄 train_models.py             # Model training script
├── 📄 test_spam_detection.py      # Testing script
│
├── 🤖 AI Models
│   ├── deep_learning_model.py     # Deep learning implementation
│   ├── main_dl.py                 # Enhanced FastAPI app
│   └── main.py                    # Original FastAPI app
│
├── 🎨 Frontend
│   ├── templates/
│   │   ├── index_dl.html          # Enhanced interface
│   │   └── index.html             # Original interface
│   └── static/
│       ├── css/styles.css         # Styling
│       ├── js/main_dl.js          # Enhanced JavaScript
│       └── js/main.js             # Original JavaScript
│
├── 📊 Data (you need to add)
│   └── mail_data.csv              # SMS spam dataset
│
└── 🔧 Generated (after training)
    ├── models/
    │   ├── spam_detector_lstm.h5
    │   ├── spam_detector_ensemble.h5
    │   ├── tokenizer_lstm.pkl
    │   └── text_classification.pkl
    └── outputs/
        ├── training_history_dl.png
        ├── confusion_matrix_dl.png
        └── model_comparison.png
```

## 🎯 Model Selection Guide

### When to Use Each Model

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Ensemble** | Production use | Highest accuracy, robust | Slower inference |
| **Deep Learning** | Complex patterns | Context understanding | Requires more resources |
| **Traditional ML** | Fast deployment | Speed, interpretability | Lower accuracy |

### Performance Comparison
```python
# Test different models
models = ['traditional', 'deep_learning', 'ensemble']
test_text = "URGENT: Verify your account now!"

for model in models:
    result = analyze_text(test_text, model)
    print(f"{model:15}: {result['prediction']:8} ({result['confidence']:5.1f}%)")
```

## 🔧 Configuration

### Model Parameters
```python
# Deep Learning Configuration
SpamDetectorDL(
    max_features=10000,    # Vocabulary size
    max_length=100,        # Sequence length  
    embedding_dim=128      # Embedding dimensions
)

# Training Parameters
epochs=30
batch_size=32
validation_split=0.2
early_stopping_patience=10

# Ensemble Weights
dl_weight = 0.7           # Deep learning model weight
traditional_weight = 0.3   # Traditional ML weight
```

### Environment Variables
```bash
# Optional: Disable TensorFlow warnings
export TF_ENABLE_ONEDNN_OPTS=0

# Optional: Set TensorFlow log level
export TF_CPP_MIN_LOG_LEVEL=2
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Test all models with various samples
python test_spam_detection.py

# Expected output:
# ✅ Obvious spam detection: 80-95%
# ✅ Legitimate messages: 95-100%  
# ⚠️  Subtle spam detection: 60-80%
```

### Manual Testing
```python
# Test specific samples
test_samples = [
    "CONGRATULATIONS! You won $1000!",           # Should be: Spam
    "Hey, lunch tomorrow at 12pm?",              # Should be: Not Spam
    "Your account has been compromised!",        # Should be: Spam
    "Meeting moved to Monday",                   # Should be: Not Spam
]

for text in test_samples:
    result = detector.predict(text)
    print(f"'{text}' → {result['prediction']} ({result['confidence']:.1f}%)")
```

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

#### Server Not Starting
```bash
# Check if port is in use
netstat -an | findstr :8000

# Use different port
uvicorn main_dl:app --port 8001
```

#### Low Accuracy
```bash
# Retrain with more epochs
python train_models.py --model deep_learning --epochs 50

# Check dataset quality
python -c "import pandas as pd; print(pd.read_csv('mail_data.csv').info())"
```

## 📚 API Reference

### Endpoints

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `GET` | `/` | Web interface | - |
| `POST` | `/api/analyze` | Analyze text | `text`, `model_type` |
| `GET` | `/api/models/compare` | Compare models | `text` |
| `GET` | `/api/models/status` | Model availability | - |
| `GET` | `/api/demo-text` | Sample text | - |

### Response Format
```json
{
  "prediction": "Spam",
  "confidence": 89.3,
  "spam_probability": 89.3,
  "not_spam_probability": 10.7,
  "model_used": "Deep Learning (LSTM/CNN)",
  "urls": [
    {
      "domain": "suspicious-site.com",
      "trust_score": 25,
      "classification": "Suspicious",
      "risk_factors": ["Contains suspicious keywords"]
    }
  ]
}
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

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

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: UCI ML Repository - SMS Spam Collection
- **Frameworks**: TensorFlow, Keras, FastAPI, scikit-learn
- **Libraries**: NLTK, pandas, numpy, matplotlib
- **UI**: Chart.js, Font Awesome, modern CSS

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/spam-text-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/spam-text-detector/discussions)
- **Documentation**: Check the `README_DL.md` for detailed technical docs

## 🔮 Future Enhancements

- [ ] **Transformer Models**: BERT/RoBERTa integration
- [ ] **Multi-language Support**: Detect spam in different languages
- [ ] **Real-time Learning**: Online learning capabilities
- [ ] **Mobile App**: React Native/Flutter implementation
- [ ] **Browser Extension**: Chrome/Firefox extension
- [ ] **Email Integration**: Gmail/Outlook plugins
- [ ] **Advanced Analytics**: Detailed reporting dashboard
- [ ] **A/B Testing**: Model performance comparison tools

---

**⭐ Star this repository if you found it helpful!**

**🛡️ Happy Spam Detecting!**