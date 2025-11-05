import numpy as np
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, 
    GlobalMaxPooling1D, Conv1D, MaxPooling1D, Input,
    Concatenate, BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpamDetectorDL:
    def __init__(self, max_features=10000, max_length=100, embedding_dim=128):
        self.max_features = max_features
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def prepare_data(self, df):
        """Prepare data for deep learning model"""
        # Preprocess text
        df['processed_text'] = df['Message'].apply(self.preprocess_text)
        
        # Convert labels to binary
        df['label'] = df['Category'].map({'spam': 1, 'ham': 0})
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['processed_text'])
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        y = df['label'].values
        
        return X, y
    
    def build_lstm_model(self):
        """Build LSTM model"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_lstm_model(self):
        """Build hybrid CNN-LSTM model"""
        model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_length),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(64, 5, activation='relu'),
            MaxPooling1D(5),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_ensemble_model(self):
        """Build ensemble model with multiple architectures"""
        input_layer = Input(shape=(self.max_length,))
        embedding = Embedding(self.max_features, self.embedding_dim)(input_layer)
        
        # LSTM branch
        lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(embedding)
        lstm_branch = GlobalMaxPooling1D()(lstm_branch)
        lstm_branch = Dense(32, activation='relu')(lstm_branch)
        
        # CNN branch
        cnn_branch = Conv1D(64, 3, activation='relu')(embedding)
        cnn_branch = GlobalMaxPooling1D()(cnn_branch)
        cnn_branch = Dense(32, activation='relu')(cnn_branch)
        
        # Combine branches
        combined = Concatenate()([lstm_branch, cnn_branch])
        combined = Dense(64, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='lstm', epochs=50, batch_size=32):
        """Train the deep learning model"""
        # Build model based on type
        if model_type == 'lstm':
            self.model = self.build_lstm_model()
        elif model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        elif model_type == 'ensemble':
            self.model = self.build_ensemble_model()
        else:
            raise ValueError("model_type must be 'lstm', 'cnn_lstm', or 'ensemble'")
        
        print(f"Training {model_type.upper()} model...")
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_dl.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred_prob
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_dl.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, texts):
        """Predict spam probability for new texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Predict
        predictions = self.model.predict(X)
        
        results = []
        for i, pred in enumerate(predictions):
            spam_prob = float(pred[0])
            ham_prob = 1 - spam_prob
            label = "Spam" if spam_prob > 0.5 else "Ham"
            confidence = max(spam_prob, ham_prob) * 100
            
            results.append({
                'text': texts[i],
                'prediction': label,
                'spam_probability': spam_prob * 100,
                'ham_probability': ham_prob * 100,
                'confidence': confidence
            })
        
        return results[0] if len(results) == 1 else results
    
    def save_model(self, model_path='spam_detector_dl.h5', tokenizer_path='tokenizer_dl.pkl'):
        """Save the trained model and tokenizer"""
        if self.model is None:
            print("No model to save")
            return
        
        # Save model
        self.model.save(model_path)
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save model parameters
        params = {
            'max_features': self.max_features,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim
        }
        with open('model_params_dl.pkl', 'wb') as f:
            pickle.dump(params, f)
        
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    @classmethod
    def load_model(cls, model_path='spam_detector_dl.h5', tokenizer_path='tokenizer_dl.pkl'):
        """Load a trained model and tokenizer"""
        # Load parameters
        with open('model_params_dl.pkl', 'rb') as f:
            params = pickle.load(f)
        
        # Create instance
        detector = cls(**params)
        
        # Load model
        detector.model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            detector.tokenizer = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        return detector

def main():
    """Main training function"""
    print("Loading data...")
    df = pd.read_csv('mail_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Category'].value_counts()}")
    
    # Initialize detector
    detector = SpamDetectorDL(max_features=10000, max_length=100, embedding_dim=128)
    
    # Prepare data
    print("\nPreparing data...")
    X, y = detector.prepare_data(df)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train different models
    models_to_train = ['lstm', 'cnn_lstm', 'ensemble']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        # Create new detector instance for each model
        detector = SpamDetectorDL(max_features=10000, max_length=100, embedding_dim=128)
        detector.tokenizer = Tokenizer(num_words=detector.max_features, oov_token='<OOV>')
        detector.tokenizer.fit_on_texts(df['Message'].apply(detector.preprocess_text))
        
        # Train model
        history = detector.train_model(X_train, y_train, X_val, y_val, 
                                     model_type=model_type, epochs=30, batch_size=32)
        
        # Evaluate model
        accuracy, predictions = detector.evaluate_model(X_test, y_test)
        results[model_type] = accuracy
        
        # Plot training history
        detector.plot_training_history()
        
        # Save model
        detector.save_model(f'spam_detector_{model_type}.h5', f'tokenizer_{model_type}.pkl')
    
    # Compare results
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    for model_type, accuracy in results.items():
        print(f"{model_type.upper()}: {accuracy:.4f}")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model.upper()} with accuracy: {results[best_model]:.4f}")
    
    # Test with sample texts
    print(f"\n{'='*50}")
    print("TESTING WITH SAMPLE TEXTS")
    print(f"{'='*50}")
    
    # Load best model for testing
    best_detector = SpamDetectorDL.load_model(f'spam_detector_{best_model}.h5', f'tokenizer_{best_model}.pkl')
    
    sample_texts = [
        "Congratulations! You've won $1000! Click here to claim your prize now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be suspended. Verify now at suspicious-link.com",
        "Thanks for the meeting today. I'll send the report by Friday.",
        "FREE MONEY! No strings attached! Call now!"
    ]
    
    for text in sample_texts:
        result = best_detector.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.1f}%)")
        print(f"Spam Probability: {result['spam_probability']:.1f}%")

if __name__ == "__main__":
    main()