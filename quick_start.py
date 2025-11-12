#!/usr/bin/env python3
"""
Quick Start Script for FinSecure AI
Verifies installation and starts the application
"""

import os
import sys
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'tensorflow',
        'sklearn',
        'pandas',
        'numpy',
        'nltk'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print_header("Checking NLTK Data")
    
    try:
        import nltk
        
        required_data = ['punkt', 'stopwords']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
                print(f"✅ {data}")
            except LookupError:
                print(f"⬇️  Downloading {data}...")
                nltk.download(data, quiet=True)
                print(f"✅ {data} downloaded")
        
        print("\n✅ NLTK data is ready")
        return True
    except Exception as e:
        print(f"❌ Error with NLTK data: {e}")
        return False

def verify_files():
    """Verify essential files are present"""
    print_header("Verifying Essential Files")
    
    result = subprocess.run([sys.executable, 'verify_files.py'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    
    return result.returncode == 0

def start_application():
    """Start the FastAPI application"""
    print_header("Starting FinSecure AI")
    
    print("\n🚀 Starting application...")
    print("📍 Access the app at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, 'main_dl.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Goodbye!")

def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🛡️  FinSecure AI - Quick Start  🛡️               ║
    ║                                                               ║
    ║     Complete Financial Security Platform                     ║
    ║     • Spam Detection (ML + DL)                               ║
    ║     • Financial Chatbot                                      ║
    ║     • Budget Tracker                                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("NLTK Data", download_nltk_data),
        ("Essential Files", verify_files)
    ]
    
    for check_name, check_func in checks:
        if not check_func():
            print(f"\n❌ {check_name} check failed!")
            print("Please fix the issues above before running the application.")
            print("Refer to SETUP_GUIDE.md for detailed instructions.")
            return 1
    
    # All checks passed
    print_header("✅ All Checks Passed!")
    print("\nYour system is ready to run FinSecure AI!")
    
    # Ask user if they want to start
    response = input("\nStart the application now? (Y/n): ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        start_application()
    else:
        print("\n👍 You can start the application later with: python main_dl.py")
        print("📖 For more information, see SETUP_GUIDE.md")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Goodbye!")
        sys.exit(0)
