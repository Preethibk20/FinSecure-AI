import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List

class FinancialChatbot:
    def __init__(self):
        self.vectorizer = None
        self.qa_data = None
        self.qa_vectors = None
        self.categories = {
            'savings': ['save', 'saving', 'savings', 'deposit', 'emergency fund'],
            'investment': ['invest', 'stock', 'bond', 'mutual fund', 'portfolio', 'return', 'dividend'],
            'debt': ['debt', 'loan', 'credit card', 'mortgage', 'interest', 'repay', 'emi'],
            'budget': ['budget', 'expense', 'spending', 'income', 'money management'],
            'retirement': ['retirement', 'pension', '401k', 'ira', 'retire'],
            'tax': ['tax', 'taxation', 'deduction', 'refund', 'filing'],
            'insurance': ['insurance', 'policy', 'premium', 'coverage', 'claim']
        }
        
    def create_training_data(self):
        """Create financial Q&A dataset"""
        qa_pairs = [
            ("How much should I save each month?", "Aim to save at least 20% of your monthly income. Start with the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt repayment."),
            ("What is an emergency fund?", "An emergency fund is savings set aside for unexpected expenses like medical bills or job loss. Aim for 3-6 months of living expenses in a liquid, accessible account."),
            ("Where should I keep my savings?", "Keep emergency savings in a high-yield savings account for easy access. For long-term goals, consider CDs, money market accounts, or investment accounts."),
            ("Should I invest in stocks?", "Stocks can offer good long-term returns but come with risk. Diversify your portfolio, invest for the long term, and only invest money you won't need soon. Consider index funds for beginners."),
            ("What is diversification?", "Diversification means spreading investments across different assets (stocks, bonds, real estate) to reduce risk. Don't put all your eggs in one basket."),
            ("How do I start investing?", "Start by setting clear goals, understanding your risk tolerance, and educating yourself. Begin with low-cost index funds or ETFs through a brokerage account or retirement plan."),
            ("What are mutual funds?", "Mutual funds pool money from many investors to buy a diversified portfolio of stocks, bonds, or other securities. They're professionally managed and good for beginners."),
            ("How do I pay off credit card debt?", "Use the avalanche method (pay highest interest first) or snowball method (pay smallest balance first). Pay more than the minimum, stop adding new debt, and consider balance transfers."),
            ("Should I pay off debt or invest?", "Generally, pay off high-interest debt (over 7-8%) first. For low-interest debt like mortgages, you might invest while making regular payments. Always maintain emergency savings."),
            ("What is a good credit score?", "Credit scores range from 300-850. Good is 670-739, very good is 740-799, and excellent is 800+. Pay bills on time, keep credit utilization low, and maintain old accounts."),
            ("How do I create a budget?", "Track all income and expenses for a month. Categorize spending, identify areas to cut, and allocate money using the 50/30/20 rule. Use apps or spreadsheets to monitor progress."),
            ("How can I reduce expenses?", "Review subscriptions, cook at home, use coupons, negotiate bills, buy generic brands, reduce energy usage, and avoid impulse purchases. Small changes add up."),
            ("What is the 50/30/20 rule?", "Allocate 50% of income to needs (housing, food, utilities), 30% to wants (entertainment, dining out), and 20% to savings and debt repayment."),
            ("When should I start saving for retirement?", "Start as early as possible to benefit from compound interest. Even small amounts in your 20s can grow significantly by retirement. Never too late to start though."),
            ("What is a 401k?", "A 401k is an employer-sponsored retirement account. Contributions are pre-tax, reducing current taxable income. Many employers match contributions - always contribute enough to get the full match."),
            ("How much do I need for retirement?", "A common rule is 25 times your annual expenses, or aim to replace 70-80% of pre-retirement income. Use retirement calculators and consider consulting a financial advisor."),
            ("How can I reduce my taxes?", "Contribute to retirement accounts (401k, IRA), use tax-advantaged accounts (HSA, FSA), claim all eligible deductions, consider tax-loss harvesting, and donate to charity."),
            ("What are tax deductions?", "Tax deductions reduce taxable income. Common ones include mortgage interest, charitable donations, student loan interest, and business expenses. Keep good records."),
            ("What insurance do I need?", "Essential: health, auto (if you drive), and renters/homeowners. Consider life insurance if others depend on your income, and disability insurance to protect earnings."),
            ("How much life insurance do I need?", "A common rule is 10-12 times your annual income. Consider your debts, dependents' needs, and future expenses like college. Term life insurance is usually most affordable."),
            ("How do I build wealth?", "Live below your means, save consistently, invest for the long term, avoid high-interest debt, increase income through skills/career growth, and stay disciplined with financial goals."),
            ("What is compound interest?", "Compound interest is earning interest on your interest. Money grows exponentially over time. Starting early makes a huge difference - it's the most powerful wealth-building tool."),
            ("Should I buy or rent a home?", "It depends on your situation. Consider: how long you'll stay (buy if 5+ years), local market prices, job stability, maintenance costs, and opportunity cost of down payment."),
            ("How do I improve my financial situation?", "Create a budget, build emergency savings, pay off high-interest debt, increase income, invest consistently, educate yourself about personal finance, and set clear financial goals."),
        ]
        
        self.qa_data = pd.DataFrame(qa_pairs, columns=['question', 'answer'])
        return self.qa_data
    
    def train(self):
        """Train the chatbot using TF-IDF vectorization"""
        if self.qa_data is None:
            self.create_training_data()
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=500
        )
        
        self.qa_vectors = self.vectorizer.fit_transform(self.qa_data['question'])
        print(f"Chatbot trained on {len(self.qa_data)} Q&A pairs")
        
    def categorize_question(self, question: str) -> str:
        """Identify the financial category of the question"""
        question_lower = question.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def get_response(self, user_question: str) -> Dict:
        """Get response for user question"""
        if self.vectorizer is None or self.qa_vectors is None:
            return {
                'answer': 'Chatbot not trained yet.',
                'confidence': 0.0,
                'category': 'error',
                'similar_questions': []
            }
        
        user_vector = self.vectorizer.transform([user_question])
        similarities = cosine_similarity(user_vector, self.qa_vectors)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_similarities = similarities[top_indices]
        
        best_idx = top_indices[0]
        confidence = float(top_similarities[0])
        category = self.categorize_question(user_question)
        
        similar_questions = []
        for idx, sim in zip(top_indices, top_similarities):
            if sim > 0.1:
                similar_questions.append({
                    'question': self.qa_data.iloc[idx]['question'],
                    'similarity': float(sim)
                })
        
        if confidence > 0.3:
            answer = self.qa_data.iloc[best_idx]['answer']
            matched_question = self.qa_data.iloc[best_idx]['question']
        else:
            answer = self._get_fallback_response(category)
            matched_question = None
        
        return {
            'answer': answer,
            'confidence': confidence,
            'category': category,
            'matched_question': matched_question,
            'similar_questions': similar_questions[:3]
        }
    
    def _get_fallback_response(self, category: str) -> str:
        """Provide fallback response when no good match found"""
        fallback_responses = {
            'savings': "For savings advice, consider building an emergency fund of 3-6 months expenses and saving at least 20% of your income.",
            'investment': "For investment advice, start with understanding your risk tolerance and consider diversified index funds for long-term growth.",
            'debt': "For debt management, prioritize high-interest debt first and consider the avalanche or snowball method for repayment.",
            'budget': "For budgeting help, try the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Track all expenses for a month to start.",
            'retirement': "For retirement planning, start saving early to benefit from compound interest. Contribute enough to get employer 401k match.",
            'tax': "For tax advice, maximize retirement contributions, keep good records of deductions, and consider consulting a tax professional.",
            'insurance': "For insurance questions, ensure you have health, auto (if applicable), and consider life insurance if others depend on your income.",
            'general': "I can help with savings, investments, debt management, budgeting, retirement, taxes, and insurance. Please ask a more specific question."
        }
        
        return fallback_responses.get(category, fallback_responses['general'])
    
    def save_model(self, filepath: str = 'financial_chatbot.pkl'):
        """Save trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'qa_data': self.qa_data,
            'qa_vectors': self.qa_vectors,
            'categories': self.categories
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'financial_chatbot.pkl'):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.qa_data = model_data['qa_data']
            self.qa_vectors = model_data['qa_vectors']
            self.categories = model_data['categories']
            
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False

if __name__ == "__main__":
    chatbot = FinancialChatbot()
    chatbot.create_training_data()
    chatbot.train()
    chatbot.save_model()
    print("Financial chatbot trained and saved!")
