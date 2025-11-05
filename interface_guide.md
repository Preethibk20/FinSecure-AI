# 🎯 Algorithm Switching & Comparison Guide

## Where to Switch Algorithms on the Localhost Page

### **📍 Location 1: Model Selection Section (Main Page)**

When you open **http://localhost:8000**, you'll see:

```
┌─────────────────────────────────────────────────────────┐
│                 🤖 Select AI Model                      │
├─────────────────────────────────────────────────────────┤
│  ○ 🔄 Ensemble                                         │
│     Best accuracy (combines all models)                │
│     ✅ Available                                        │
│                                                         │
│  ○ 🧠 Deep Learning                                    │
│     LSTM/CNN neural networks                           │
│     ✅ Available                                        │
│                                                         │
│  ○ 🧮 Traditional ML                                   │
│     Naive Bayes with TF-IDF                           │
│     ✅ Available                                        │
└─────────────────────────────────────────────────────────┘
```

**How to use:**
1. Click on any radio button to select the algorithm
2. Enter your text in the text area below
3. Click "🔍 Analyze Text"
4. See results with the selected model

---

### **📍 Location 2: Compare All Models Button**

Below the text input area, you'll find:

```
┌─────────────────────────────────────────────────────────┐
│  [🔍 Analyze Text]  [🗑️ Clear]  [⚖️ Compare All Models] │
└─────────────────────────────────────────────────────────┘
```

**How to use:**
1. Enter text in the input area
2. Click "⚖️ Compare All Models"
3. See side-by-side comparison of all three algorithms

---

### **📍 Location 3: Model Comparison Tab**

At the top of the page, you'll see tabs:

```
┌─────────────────────────────────────────────────────────┐
│ [🔍 Spam Detection] [⚖️ Model Comparison] [⚙️ How It Works] │
└─────────────────────────────────────────────────────────┘
```

**How to use:**
1. Click on "⚖️ Model Comparison" tab
2. Enter text in the comparison text area
3. Click "⚖️ Compare All Models"
4. See detailed comparison results

---

## 🔍 What You'll See When Comparing Models

### **Individual Model Results:**
```
🤖 TRADITIONAL ML
   Prediction: Not Spam
   Confidence: 62.1%
   Spam Prob: 37.9%
   Model: Traditional ML (Naive Bayes)

🧠 DEEP LEARNING  
   Prediction: Spam
   Confidence: 89.3%
   Spam Prob: 89.3%
   Model: Deep Learning (LSTM/CNN)

🔄 ENSEMBLE
   Prediction: Spam
   Confidence: 78.5%
   Spam Prob: 78.5%
   Model: Ensemble (Traditional ML + Deep Learning)
```

### **Ensemble Breakdown (when using Ensemble):**
```
🔄 Ensemble Model Breakdown

Traditional ML (Weight: 30%)    Deep Learning (Weight: 70%)
├─ Prediction: Not Spam         ├─ Prediction: Spam
├─ Confidence: 62.1%            ├─ Confidence: 89.3%
└─ Spam: 37.9% | Ham: 62.1%     └─ Spam: 89.3% | Ham: 10.7%
```

---

## 🧪 Test Different Algorithms

### **Sample Texts to Try:**

**Obvious Spam:**
```
CONGRATULATIONS! You've WON $10,000 CASH! Call 1-800-WIN-NOW to claim your prize!
```

**Subtle Spam:**
```
Your package delivery failed. Update your address to reschedule delivery.
```

**Legitimate Message:**
```
Hey, are we still meeting for lunch tomorrow at 12pm?
```

### **Expected Differences:**

1. **Traditional ML (Naive Bayes):**
   - Fast but sometimes misses context
   - Good with obvious spam keywords
   - May struggle with subtle phishing

2. **Deep Learning (LSTM/CNN):**
   - Better context understanding
   - Catches subtle patterns
   - Higher accuracy on complex cases

3. **Ensemble:**
   - Best overall accuracy
   - Combines strengths of both
   - More reliable predictions

---

## 🎯 Quick Testing Steps

1. **Open:** http://localhost:8000
2. **Select Model:** Choose radio button (Ensemble recommended)
3. **Enter Text:** Type or paste your message
4. **Analyze:** Click "🔍 Analyze Text"
5. **Compare:** Click "⚖️ Compare All Models" to see differences
6. **Switch:** Try different models and compare results

---

## 📊 Performance Indicators

Look for these indicators to understand model performance:

- **Confidence Score:** Higher = more certain
- **Spam Probability:** 0-100% likelihood of spam
- **Model Used:** Shows which algorithm made the prediction
- **Ensemble Breakdown:** Shows how different models voted

The interface will show you real-time differences between algorithms, helping you understand which approach works best for different types of messages!