#!/usr/bin/env python3
"""
Test script to verify spam detection accuracy with various sample texts
"""

import requests
import json
from typing import List, Dict

class SpamDetectionTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_sample(self, text: str, model_type: str = "ensemble") -> Dict:
        """Test a single text sample"""
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json={"text": text, "model_type": model_type},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def compare_models(self, text: str) -> Dict:
        """Compare all models on a single text"""
        try:
            response = requests.get(
                f"{self.base_url}/api/models/compare",
                params={"text": text},
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def run_comprehensive_test(self):
        """Run comprehensive tests with various spam and ham samples"""
        
        # Test samples - mix of obvious spam, subtle spam, and legitimate messages
        test_samples = {
            "obvious_spam": [
                "CONGRATULATIONS! You've WON $10,000 CASH! Call 1-800-WIN-NOW to claim your prize!",
                "FREE MONEY! No strings attached! Click here: http://free-money-now.com",
                "URGENT: Your account will be SUSPENDED! Verify immediately at http://verify-account-now.com",
                "You have been selected to receive a FREE iPhone 15! Text CLAIM to 12345",
                "WINNER! You've won a $500 Walmart gift card! Click to claim: http://walmart-winner.net"
            ],
            
            "subtle_spam": [
                "Hi, I noticed you might be interested in making extra income from home. Would you like to learn more?",
                "Your prescription is ready for pickup. Confirm your details at our secure portal.",
                "Limited time offer: 50% off all items. Shop now before it's too late!",
                "Your package delivery failed. Update your address to reschedule delivery.",
                "Security alert: Unusual activity detected on your account. Please verify your identity."
            ],
            
            "legitimate_messages": [
                "Hey, are we still meeting for lunch tomorrow at 12pm?",
                "Thanks for the meeting today. I'll send you the report by Friday.",
                "Happy birthday! Hope you have a wonderful day!",
                "The project deadline has been moved to next Monday. Please update your schedules.",
                "Reminder: Your dentist appointment is scheduled for tomorrow at 2pm.",
                "Can you pick up milk on your way home? We're running low.",
                "The weather looks great for our picnic this weekend!",
                "Your order #12345 has been shipped and will arrive in 2-3 business days."
            ],
            
            "edge_cases": [
                "Free delivery on orders over $50. Valid until Sunday.",  # Promotional but legitimate
                "Your bank account has been compromised. Call us immediately.",  # Phishing attempt
                "Meeting cancelled due to weather. Will reschedule soon.",  # Short legitimate message
                "SALE SALE SALE! Everything must go! 90% off!",  # Aggressive marketing
                "Your verification code is 123456. Do not share this code.",  # Legitimate security message
                "",  # Empty message
                "Ok",  # Very short message
                "https://suspicious-link.com/claim-prize-now"  # Just a suspicious URL
            ]
        }
        
        print("=" * 80)
        print("COMPREHENSIVE SPAM DETECTION TEST")
        print("=" * 80)
        
        all_results = {}
        
        for category, messages in test_samples.items():
            print(f"\n📂 Testing {category.upper().replace('_', ' ')} ({len(messages)} samples)")
            print("-" * 60)
            
            category_results = []
            
            for i, message in enumerate(messages, 1):
                print(f"\n{i}. Testing: {message[:60]}{'...' if len(message) > 60 else ''}")
                
                # Test with ensemble model
                result = self.test_sample(message, "ensemble")
                
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                    continue
                
                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0)
                spam_prob = result.get("spam_probability", 0)
                model_used = result.get("model_used", "Unknown")
                
                # Determine if prediction seems correct
                expected_spam = category in ["obvious_spam", "subtle_spam"] or (
                    category == "edge_cases" and any(word in message.lower() for word in 
                    ["compromised", "suspicious-link", "claim-prize"])
                )
                
                is_correct = (prediction == "Spam") == expected_spam
                status_icon = "✅" if is_correct else "❌"
                
                print(f"   {status_icon} Prediction: {prediction}")
                print(f"      Confidence: {confidence:.1f}%")
                print(f"      Spam Probability: {spam_prob:.1f}%")
                print(f"      Model: {model_used}")
                
                category_results.append({
                    "message": message,
                    "prediction": prediction,
                    "confidence": confidence,
                    "spam_probability": spam_prob,
                    "model_used": model_used,
                    "expected_spam": expected_spam,
                    "correct": is_correct
                })
            
            all_results[category] = category_results
            
            # Category summary
            correct_predictions = sum(1 for r in category_results if r["correct"])
            total_predictions = len(category_results)
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            print(f"\n📊 {category.upper()} SUMMARY:")
            print(f"   Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        # Overall summary
        self.print_overall_summary(all_results)
        
        return all_results
    
    def print_overall_summary(self, results: Dict):
        """Print overall test summary"""
        print("\n" + "=" * 80)
        print("OVERALL TEST SUMMARY")
        print("=" * 80)
        
        total_correct = 0
        total_tests = 0
        
        for category, category_results in results.items():
            correct = sum(1 for r in category_results if r["correct"])
            total = len(category_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            total_correct += correct
            total_tests += total
            
            print(f"{category.replace('_', ' ').title():20}: {correct:2}/{total:2} ({accuracy:5.1f}%)")
        
        overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
        print("-" * 40)
        print(f"{'Overall Accuracy':20}: {total_correct:2}/{total_tests:2} ({overall_accuracy:5.1f}%)")
        
        # Performance interpretation
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        if overall_accuracy >= 90:
            print("   🎉 Excellent performance! The model is working very well.")
        elif overall_accuracy >= 80:
            print("   👍 Good performance! The model is working well with minor issues.")
        elif overall_accuracy >= 70:
            print("   ⚠️  Moderate performance. The model needs some improvement.")
        else:
            print("   🚨 Poor performance. The model may need retraining or debugging.")
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TEST")
        print("=" * 80)
        
        test_message = "URGENT: Your account has been compromised! Click here to verify: http://verify-now.com"
        
        print(f"Testing message: {test_message}")
        print("-" * 60)
        
        comparison_result = self.compare_models(test_message)
        
        if "error" in comparison_result:
            print(f"❌ Error: {comparison_result['error']}")
            return
        
        predictions = comparison_result.get("model_predictions", {})
        
        for model_name, result in predictions.items():
            print(f"\n🤖 {model_name.upper().replace('_', ' ')}:")
            print(f"   Prediction: {result.get('prediction', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.1f}%")
            print(f"   Spam Prob: {result.get('spam_probability', 0):.1f}%")
            print(f"   Model: {result.get('model_used', 'Unknown')}")

def main():
    """Main test function"""
    print("🧪 Starting Spam Detection Tests...")
    
    tester = SpamDetectionTester()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/models/status", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly. Please ensure the application is running.")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please ensure the application is running on http://localhost:8000")
        return
    
    print("✅ Server is running. Starting tests...\n")
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    # Test model comparison
    tester.test_model_comparison()
    
    print(f"\n🎯 Test completed! Check the results above to verify the model's performance.")
    print(f"💡 If accuracy is low, consider retraining the models with more data.")

if __name__ == "__main__":
    main()