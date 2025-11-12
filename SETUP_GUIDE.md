# 🚀 FinSecure AI - Setup Guide

## Essential Files for Running the Application

This document lists all the files required to run the FinSecure AI application on any computer after cloning.

### ✅ Core Application Files

#### Python Backend
- `main_dl.py` - Main FastAPI application (enhanced version with all features)
- `main.py` - Original spam detector only version
- `deep_learning_model.py` - Deep learning model implementation
- `financial_chatbot.py` - Financial chatbot implementation

#### HTML Templates (templates/)
- `home.html` - Landing page with all features
- `index_dl.html` - Advanced spam detector page
- `index.html` - Original spam detector page
- `financial_chatbot.html` - Financial chatbot page
- `budget_tracker.html` - Budget tracker page

#### Static Assets (static/)
- `static/css/styles.css` - All styling
- `static/js/main_dl.js` - JavaScript for advanced spam detector
- `static/js/main.js` - JavaScript for original spam detector

### 🤖 Pre-trained Models (REQUIRED)

#### Traditional ML Model
- `text_classification.pkl` - Trained Naive Bayes model with TF-IDF vectorizer

#### Deep Learning Models
- `spam_detector_lstm.h5` - LSTM neural network model
- `spam_detector_cnn_lstm.h5` - CNN-LSTM hybrid model
- `spam_detector_ensemble.h5` - Ensemble model (best performance)

#### Tokenizers
- `tokenizer_lstm.pkl` - Tokenizer for LSTM model
- `tokenizer_cnn_lstm.pkl` - Tokenizer for CNN-LSTM model
- `tokenizer_ensemble.pkl` - Tokenizer for ensemble model
- `model_params_dl.pkl` - Model parameters (max_features, max_length, embedding_dim)

#### Financial Chatbot Model
- `financial_chatbot.pkl` - Trained financial advice chatbot model

### 📊 Data Files (REQUIRED)
- `mail_data.csv` - SMS spam dataset (needed for retraining only, not for running)

### 📝 Configuration Files
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

### ❌ Files NOT Needed for Running (Can be deleted)
- `train_models.py` - Only needed for retraining models
- `confusion_matrix_dl.png` - Training visualization
- `training_history_dl.png` - Training visualization
- `SMS-CLASSIFIER-main/` - Empty folder
- `__pycache__/` - Python cache (auto-generated)
- `.vscode/` - Editor settings
- `venv/` - Virtual environment (recreate on each machine)

## 📦 Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (Required for Deep Learning)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 5. Verify All Models Exist
```bash
# Check if all required model files exist
python -c "import os; required = ['text_classification.pkl', 'spam_detector_ensemble.h5', 'tokenizer_ensemble.pkl', 'model_params_dl.pkl', 'financial_chatbot.pkl']; missing = [f for f in required if not os.path.exists(f)]; print('All models present!' if not missing else f'Missing: {missing}')"
```

### 6. Run the Application
```bash
# Full application with all features
python main_dl.py

# Or just spam detector
python main.py
```

### 7. Access the Application
Open your browser and go to:
- **Full App**: http://localhost:8000
- **Spam Detector**: http://localhost:8000/spam-detector
- **Financial Chatbot**: http://localhost:8000/financial-chatbot
- **Budget Tracker**: http://localhost:8000/budget-tracker

## 🔍 Verify Installation

### Check Model Status
```bash
curl http://localhost:8000/api/models/status
```

Expected response:
```json
{
  "traditional_ml": {"available": true, "type": "Naive Bayes with TF-IDF"},
  "deep_learning": {"available": true, "type": "LSTM/CNN Neural Network"},
  "ensemble": {"available": true, "type": "Weighted combination of both models"}
}
```

### Test Spam Detection
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: You won $1000!", "model_type": "ensemble"}'
```

### Test Financial Chatbot
```bash
curl -X POST http://localhost:8000/api/financial-advice \
  -H "Content-Type: application/json" \
  -d '{"question": "How much should I save each month?"}'
```

## 📋 File Checklist

Before sharing your project, ensure these files are present:

### Must Have ✅
- [ ] `main_dl.py`
- [ ] `main.py`
- [ ] `deep_learning_model.py`
- [ ] `financial_chatbot.py`
- [ ] `requirements.txt`
- [ ] `README.md`
- [ ] All templates (5 HTML files)
- [ ] All static files (CSS, JS)
- [ ] `text_classification.pkl`
- [ ] `spam_detector_ensemble.h5` (or lstm/cnn_lstm)
- [ ] `tokenizer_ensemble.pkl` (or lstm/cnn_lstm)
- [ ] `model_params_dl.pkl`
- [ ] `financial_chatbot.pkl`

### Optional 📝
- [ ] `mail_data.csv` (only if you want others to retrain)
- [ ] `SETUP_GUIDE.md` (this file)

### Can Delete ❌
- [ ] `train_models.py`
- [ ] `*.png` (training visualizations)
- [ ] `SMS-CLASSIFIER-main/` (empty folder)
- [ ] `__pycache__/`
- [ ] `.vscode/`
- [ ] `venv/`

## 🐛 Troubleshooting

### Models Not Loading
```bash
# Check if model files exist
ls -la *.pkl *.h5

# Recreate model_params_dl.pkl if missing
python -c "import pickle; params = {'max_features': 10000, 'max_length': 100, 'embedding_dim': 128}; pickle.dump(params, open('model_params_dl.pkl', 'wb'))"
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Port Already in Use
```bash
# Use different port
python main_dl.py --port 8001

# Or in code, change:
# uvicorn.run("main_dl:app", host="0.0.0.0", port=8001, reload=True)
```

## 📦 Creating a Distribution Package

To share your project with all necessary files:

```bash
# Create a clean distribution folder
mkdir finsecure-ai-dist
cd finsecure-ai-dist

# Copy essential files
cp ../main_dl.py .
cp ../main.py .
cp ../deep_learning_model.py .
cp ../financial_chatbot.py .
cp ../requirements.txt .
cp ../README.md .
cp ../SETUP_GUIDE.md .

# Copy models
cp ../*.pkl .
cp ../*.h5 .

# Copy templates and static folders
cp -r ../templates .
cp -r ../static .

# Optional: Include dataset
cp ../mail_data.csv .

# Create zip
cd ..
zip -r finsecure-ai.zip finsecure-ai-dist/
```

## 🎯 Minimum System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for deep learning)
- **Disk Space**: 500MB for dependencies + models
- **OS**: Windows, Linux, or macOS

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify all required files are present
3. Check Python version: `python --version`
4. Check installed packages: `pip list`
5. Review error messages carefully

---

**✅ Once setup is complete, you're ready to use FinSecure AI!**
