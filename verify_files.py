#!/usr/bin/env python3
"""
Verify that all essential files are present for running FinSecure AI
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {filepath:40} - {description}")
    return exists

def check_directory(dirpath, description):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {dirpath:40} - {description}")
    return exists

def main():
    print("=" * 80)
    print("FinSecure AI - File Verification")
    print("=" * 80)
    
    all_present = True
    
    # Core Python files
    print("\n📄 Core Python Files:")
    all_present &= check_file("main_dl.py", "Main application (enhanced)")
    all_present &= check_file("main.py", "Original spam detector")
    all_present &= check_file("deep_learning_model.py", "DL model implementation")
    all_present &= check_file("financial_chatbot.py", "Financial chatbot")
    all_present &= check_file("requirements.txt", "Dependencies")
    
    # Templates
    print("\n📝 HTML Templates:")
    all_present &= check_directory("templates", "Templates folder")
    all_present &= check_file("templates/home.html", "Landing page")
    all_present &= check_file("templates/index_dl.html", "Advanced spam detector")
    all_present &= check_file("templates/index.html", "Original spam detector")
    all_present &= check_file("templates/financial_chatbot.html", "Financial chatbot page")
    all_present &= check_file("templates/budget_tracker.html", "Budget tracker page")
    
    # Static files
    print("\n🎨 Static Assets:")
    all_present &= check_directory("static", "Static folder")
    all_present &= check_directory("static/css", "CSS folder")
    all_present &= check_directory("static/js", "JavaScript folder")
    all_present &= check_file("static/css/styles.css", "Main stylesheet")
    all_present &= check_file("static/js/main_dl.js", "Enhanced JS")
    all_present &= check_file("static/js/main.js", "Original JS")
    
    # Traditional ML Model
    print("\n🤖 Traditional ML Model:")
    all_present &= check_file("text_classification.pkl", "Naive Bayes model")
    
    # Deep Learning Models
    print("\n🧠 Deep Learning Models:")
    has_lstm = check_file("spam_detector_lstm.h5", "LSTM model")
    has_cnn_lstm = check_file("spam_detector_cnn_lstm.h5", "CNN-LSTM model")
    has_ensemble = check_file("spam_detector_ensemble.h5", "Ensemble model (recommended)")
    
    if not (has_lstm or has_cnn_lstm or has_ensemble):
        print("   ⚠️  WARNING: No deep learning models found!")
        all_present = False
    
    # Tokenizers
    print("\n🔤 Tokenizers:")
    check_file("tokenizer_lstm.pkl", "LSTM tokenizer")
    check_file("tokenizer_cnn_lstm.pkl", "CNN-LSTM tokenizer")
    check_file("tokenizer_ensemble.pkl", "Ensemble tokenizer")
    all_present &= check_file("model_params_dl.pkl", "Model parameters")
    
    # Financial Chatbot
    print("\n💰 Financial Chatbot:")
    all_present &= check_file("financial_chatbot.pkl", "Chatbot model")
    
    # Optional files
    print("\n📊 Optional Files:")
    check_file("mail_data.csv", "Dataset (for retraining)")
    check_file("README.md", "Documentation")
    check_file("SETUP_GUIDE.md", "Setup guide")
    
    # Summary
    print("\n" + "=" * 80)
    if all_present:
        print("✅ SUCCESS: All essential files are present!")
        print("   You can run the application with: python main_dl.py")
        return 0
    else:
        print("❌ ERROR: Some essential files are missing!")
        print("   Please check the missing files above.")
        print("   Refer to SETUP_GUIDE.md for more information.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
