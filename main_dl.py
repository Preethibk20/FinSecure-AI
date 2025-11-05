from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pickle
import re
import numpy as np
from urllib.parse import urlparse
import uvicorn
import tensorflow as tf
from deep_learning_model import SpamDetectorDL
import os

app = FastAPI(
    title="Advanced Spam Text Detector API",
    description="API for detecting spam texts using both traditional ML and Deep Learning models",
    version="2.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global model variables
traditional_model = None
dl_model = None

def load_traditional_model():
    """Load the traditional ML model"""
    global traditional_model
    if traditional_model is None:
        try:
            with open('text_classification.pkl', 'rb') as file:
                traditional_model = pickle.load(file)
        except FileNotFoundError:
            print("Traditional model not found")
            return None
        except Exception as e:
            print(f"Error loading traditional model: {e}")
            return None
    return traditional_model

def load_dl_model():
    """Load the deep learning model"""
    global dl_model
    if dl_model is None:
        try:
            # Try to load the best performing model (you can change this based on your results)
            model_files = [
                ('spam_detector_ensemble.h5', 'tokenizer_ensemble.pkl'),
                ('spam_detector_lstm.h5', 'tokenizer_lstm.pkl'),
                ('spam_detector_cnn_lstm.h5', 'tokenizer_cnn_lstm.pkl')
            ]
            
            for model_file, tokenizer_file in model_files:
                if os.path.exists(model_file) and os.path.exists(tokenizer_file):
                    dl_model = SpamDetectorDL.load_model(model_file, tokenizer_file)
                    print(f"Loaded deep learning model: {model_file}")
                    break
            
            if dl_model is None:
                print("No deep learning model found")
                
        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            return None
    return dl_model

class TextInput(BaseModel):
    text: str
    model_type: Optional[str] = "ensemble"  # "traditional", "deep_learning", or "ensemble"

class URLAnalysisResult(BaseModel):
    domain: str
    url: str
    trust_score: float
    classification: str
    risk_factors: List[str]
    security_features: List[str]
    error: Optional[str] = None

class SpamAnalysisResult(BaseModel):
    prediction: str
    confidence: float
    spam_probability: float
    not_spam_probability: float
    model_used: str
    urls: List[URLAnalysisResult] = []
    ensemble_results: Optional[Dict[str, Any]] = None

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index_dl.html", {"request": request})

@app.post("/api/analyze", response_model=SpamAnalysisResult)
async def analyze_text(text_input: TextInput):
    text = text_input.text
    model_type = text_input.model_type
    
    # Extract and analyze URLs
    urls = extract_urls(text)
    url_results = []
    for url in urls:
        url_results.append(check_website_trust(url))
    
    # Analyze with requested model type
    if model_type == "traditional":
        result = analyze_with_traditional_model(text)
    elif model_type == "deep_learning":
        result = analyze_with_dl_model(text)
    elif model_type == "ensemble":
        result = analyze_with_ensemble(text)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid model_type. Use 'traditional', 'deep_learning', or 'ensemble'"}
        )
    
    if result is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model analysis failed"}
        )
    
    # Add URL results
    result["urls"] = url_results
    
    return result

def analyze_with_traditional_model(text):
    """Analyze text with traditional ML model"""
    model = load_traditional_model()
    if model is None:
        return None
    
    try:
        prediction = model.predict([text])
        try:
            prob = model.predict_proba([text])[0]
            spam_prob = float(prob[0] * 100)
            not_spam_prob = float(prob[1] * 100)
            confidence = float(max(prob) * 100)
        except:
            spam_prob = 85.0 if prediction[0] == 0 else 15.0
            not_spam_prob = 15.0 if prediction[0] == 0 else 85.0
            confidence = max(spam_prob, not_spam_prob)
        
        result_label = "Spam" if prediction[0] == 0 else "Not Spam"
        
        return {
            "prediction": result_label,
            "confidence": confidence,
            "spam_probability": spam_prob,
            "not_spam_probability": not_spam_prob,
            "model_used": "Traditional ML (Naive Bayes)"
        }
    except Exception as e:
        print(f"Traditional model prediction error: {e}")
        return None

def analyze_with_dl_model(text):
    """Analyze text with deep learning model"""
    model = load_dl_model()
    if model is None:
        return None
    
    try:
        result = model.predict(text)
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "spam_probability": result["spam_probability"],
            "not_spam_probability": result["ham_probability"],
            "model_used": "Deep Learning (LSTM/CNN)"
        }
    except Exception as e:
        print(f"Deep learning model prediction error: {e}")
        return None

def analyze_with_ensemble(text):
    """Analyze text with ensemble of both models"""
    traditional_result = analyze_with_traditional_model(text)
    dl_result = analyze_with_dl_model(text)
    
    # If only one model is available, use that
    if traditional_result is None and dl_result is None:
        return None
    elif traditional_result is None:
        dl_result["model_used"] = "Deep Learning Only (Traditional model unavailable)"
        return dl_result
    elif dl_result is None:
        traditional_result["model_used"] = "Traditional ML Only (DL model unavailable)"
        return traditional_result
    
    # Ensemble prediction using weighted average
    # Give more weight to deep learning model (you can adjust these weights)
    dl_weight = 0.7
    traditional_weight = 0.3
    
    # Calculate ensemble probabilities
    ensemble_spam_prob = (
        dl_result["spam_probability"] * dl_weight + 
        traditional_result["spam_probability"] * traditional_weight
    )
    ensemble_not_spam_prob = 100 - ensemble_spam_prob
    
    # Determine final prediction
    final_prediction = "Spam" if ensemble_spam_prob > 50 else "Not Spam"
    final_confidence = max(ensemble_spam_prob, ensemble_not_spam_prob)
    
    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "spam_probability": ensemble_spam_prob,
        "not_spam_probability": ensemble_not_spam_prob,
        "model_used": "Ensemble (Traditional ML + Deep Learning)",
        "ensemble_results": {
            "traditional_ml": traditional_result,
            "deep_learning": dl_result,
            "weights": {
                "deep_learning": dl_weight,
                "traditional_ml": traditional_weight
            }
        }
    }

def extract_urls(text):
    """Extract URLs from text"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[\w\d._~:/?#\[\]@!\'()*+,;=-]+'
    urls = re.findall(url_pattern, text)
    return urls

def check_website_trust(url):
    """Check website trust score (same as original implementation)"""
    try:
        domain = urlparse(url).netloc
        
        trust_score = 0
        risk_factors = []
        security_features = []
        
        established_domains = ["google.com", "microsoft.com", "amazon.com", "github.com", 
                              "apple.com", "netflix.com", "yahoo.com", "linkedin.com"]
        suspicious_keywords = ["free", "win", "prize", "crypto", "urgent", "lucky", "discount", "act-now"]
        security_keywords = ["secure", "official", "verified", "authentic"]
        
        if any(domain.endswith(ed) or ed in domain for ed in established_domains):
            trust_score = 90
            security_features.append("Established domain")
        elif len(domain) > 30:
            trust_score = 30
            risk_factors.append("Unusually long domain name")
        elif any(keyword in domain.lower() for keyword in suspicious_keywords):
            trust_score = 40
            risk_factors.append("Contains suspicious keywords")
            
            for keyword in suspicious_keywords:
                if keyword in domain.lower():
                    risk_factors.append(f"Contains '{keyword}'")
        else:
            trust_score = 65
        
        if domain.count('.') > 2:
            trust_score -= 15
            risk_factors.append("Multiple subdomains")
        
        if any(keyword in domain.lower() for keyword in security_keywords):
            if trust_score < 70:
                trust_score += 10
                security_features.append("Security signaling terms")
        
        trust_score = max(0, min(100, trust_score))
            
        return {
            "domain": domain,
            "url": url,
            "trust_score": trust_score,
            "classification": get_trust_classification(trust_score),
            "risk_factors": risk_factors,
            "security_features": security_features
        }
    except Exception as e:
        return {
            "domain": url,
            "url": url,
            "trust_score": 0,
            "classification": "Error analyzing URL",
            "risk_factors": ["Error in analysis"],
            "security_features": [],
            "error": str(e)
        }

def get_trust_classification(score):
    """Get trust classification based on score"""
    if score >= 80:
        return "High Trust"
    elif score >= 60:
        return "Moderate Trust"
    elif score >= 40:
        return "Low Trust"
    else:
        return "Suspicious"

@app.get("/api/demo-text")
async def get_demo_text():
    """Get demo text for testing"""
    demo_text = """URGENT: Your account has been compromised! 
    Click here to verify: http://secur1ty-verify.prize-winner.com
    Also check our legitimate site at https://google.com for more information.
    You've won $1000 - claim at https://free-prizes-winner.net now!"""
    
    return {"text": demo_text}

@app.get("/api/models/status")
async def get_models_status():
    """Get status of available models"""
    traditional_available = load_traditional_model() is not None
    dl_available = load_dl_model() is not None
    
    return {
        "traditional_ml": {
            "available": traditional_available,
            "type": "Naive Bayes with TF-IDF"
        },
        "deep_learning": {
            "available": dl_available,
            "type": "LSTM/CNN Neural Network"
        },
        "ensemble": {
            "available": traditional_available and dl_available,
            "type": "Weighted combination of both models"
        }
    }

@app.get("/api/models/compare")
async def compare_models(text: str):
    """Compare predictions from all available models"""
    results = {}
    
    # Traditional ML
    traditional_result = analyze_with_traditional_model(text)
    if traditional_result:
        results["traditional_ml"] = traditional_result
    
    # Deep Learning
    dl_result = analyze_with_dl_model(text)
    if dl_result:
        results["deep_learning"] = dl_result
    
    # Ensemble
    ensemble_result = analyze_with_ensemble(text)
    if ensemble_result:
        results["ensemble"] = ensemble_result
    
    return {
        "input_text": text,
        "model_predictions": results
    }

if __name__ == "__main__":
    uvicorn.run("main_dl:app", host="0.0.0.0", port=8000, reload=True)