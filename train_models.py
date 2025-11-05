#!/usr/bin/env python3
"""
Training script for spam detection models
Trains both traditional ML and deep learning models
"""

import os
import sys
import argparse
from deep_learning_model import main as train_dl_models

def train_traditional_model():
    """Train the traditional ML model using the existing notebook logic"""
    print("Training Traditional ML Model...")
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    import pickle
    
    # Load data
    try:
        mail_data = pd.read_csv('mail_data.csv')
    except FileNotFoundError:
        print("Error: mail_data.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        return False
    
    print(f"Dataset loaded: {mail_data.shape[0]} samples")
    print(f"Class distribution:\n{mail_data['Category'].value_counts()}")
    
    # Prepare data
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    
    X = mail_data['Message']
    Y = mail_data['Category'].astype('int')
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)),
        ('classifier', MultinomialNB())
    ])
    
    # Train model
    print("Training Naive Bayes model...")
    pipeline.fit(X_train, Y_train)
    
    # Evaluate
    train_accuracy = pipeline.score(X_train, Y_train)
    test_accuracy = pipeline.score(X_test, Y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions for detailed metrics
    Y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, target_names=['Spam', 'Ham']))
    
    # Save model
    with open('text_classification.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Traditional ML model saved as 'text_classification.pkl'")
    
    # Test with sample
    sample_texts = [
        "Congratulations! You've won $1000! Click here now!",
        "Hey, are we still meeting for lunch tomorrow?"
    ]
    
    print("\nTesting with sample texts:")
    for text in sample_texts:
        prediction = pipeline.predict([text])[0]
        prob = pipeline.predict_proba([text])[0]
        label = "Spam" if prediction == 0 else "Ham"
        confidence = max(prob) * 100
        print(f"Text: {text}")
        print(f"Prediction: {label} (Confidence: {confidence:.1f}%)")
        print()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train spam detection models')
    parser.add_argument('--model', choices=['traditional', 'deep_learning', 'both'], 
                       default='both', help='Which model(s) to train')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of epochs for deep learning training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPAM DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('mail_data.csv'):
        print("Error: mail_data.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        sys.exit(1)
    
    success = True
    
    if args.model in ['traditional', 'both']:
        print("\n" + "=" * 40)
        print("TRAINING TRADITIONAL ML MODEL")
        print("=" * 40)
        success &= train_traditional_model()
    
    if args.model in ['deep_learning', 'both']:
        print("\n" + "=" * 40)
        print("TRAINING DEEP LEARNING MODELS")
        print("=" * 40)
        try:
            # Modify the main function to accept epochs parameter
            import deep_learning_model
            # You might need to modify the main function to accept parameters
            train_dl_models()
            print("Deep learning models training completed!")
        except Exception as e:
            print(f"Error training deep learning models: {e}")
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run the application with:")
        print("python main_dl.py")
        print("\nOr the original version with:")
        print("python main.py")
    else:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()