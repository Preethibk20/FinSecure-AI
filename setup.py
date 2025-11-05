#!/usr/bin/env python3
"""
Setup script for the Advanced Spam Text Detector
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Download NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ NLTK data download failed: {e}")
        return False
    
    return True

def check_dataset():
    """Check if dataset exists"""
    if not os.path.exists('mail_data.csv'):
        print("\n⚠️  Dataset 'mail_data.csv' not found!")
        print("Please ensure you have the spam dataset in the current directory.")
        print("You can download it from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
        return False
    
    print("✓ Dataset 'mail_data.csv' found")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['static/css', 'static/js', 'templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup Advanced Spam Text Detector')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training (use existing models)')
    parser.add_argument('--train-only', choices=['traditional', 'deep_learning', 'both'], 
                       default='both', help='Which models to train')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADVANCED SPAM TEXT DETECTOR SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Setup incomplete - dataset missing")
        print("You can still run the application, but model training will fail.")
        print("Download the dataset and run 'python train_models.py' to train models.")
    
    # Train models (if not skipped)
    if not args.skip_training and os.path.exists('mail_data.csv'):
        print(f"\nTraining models ({args.train_only})...")
        train_command = f"python train_models.py --model {args.train_only}"
        
        if not run_command(train_command, f"Training {args.train_only} model(s)"):
            print("\n⚠️  Model training failed, but setup can continue")
            print("You can train models later using: python train_models.py")
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. If you haven't trained models yet:")
    print("   python train_models.py")
    print("\n2. Start the application:")
    print("   python main_dl.py")
    print("\n3. Open your browser and go to:")
    print("   http://localhost:8000")
    
    print("\nAvailable applications:")
    print("- Advanced version (with deep learning): python main_dl.py")
    print("- Original version (traditional ML only): python main.py")
    
    print("\nFor help:")
    print("- Training: python train_models.py --help")
    print("- Setup: python setup.py --help")

if __name__ == "__main__":
    main()