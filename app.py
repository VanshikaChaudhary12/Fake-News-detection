"""
AI-Based Fake News Detection - Flask Web Application
==================================================
This Flask app provides a web interface for fake news detection
using the trained Logistic Regression model.

Author: B.Tech CSE Student
Project: Final Year Project - Fake News Detection
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class NewsPredictor:
    def __init__(self):
        """
        Load the trained model and vectorizer for predictions
        """
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.load_model()
    
    def load_model(self):
        """
        Load the saved model and vectorizer
        """
        try:
            # Load the trained Logistic Regression model
            with open('model/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the TF-IDF vectorizer
            with open('model/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print("Model and vectorizer loaded successfully!")
            
        except FileNotFoundError:
            print("Error: Model files not found!")
            print("Please run 'python model/train_model.py' first to train the model.")
            self.model = None
            self.vectorizer = None
    
    def preprocess_text(self, text):
        """
        Preprocess text using the same steps as training
        
        This ensures consistency between training and prediction
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text):
        """
        Predict if the given text is fake or real news
        
        Returns:
        - prediction: 0 (fake) or 1 (real)
        - confidence: probability score
        - label: human-readable label
        """
        if not self.model or not self.vectorizer:
            return None, 0, "Model not loaded"
        
        if not text or text.strip() == "":
            return None, 0, "Empty text"
        
        # Preprocess the input text
        cleaned_text = self.preprocess_text(text)
        
        if cleaned_text == "":
            return None, 0, "No valid content after preprocessing"
        
        # Convert to TF-IDF features
        text_tfidf = self.vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Get confidence score (probability of predicted class)
        confidence = probabilities[prediction]
        
        # Convert to human-readable label
        label = "Real News" if prediction == 1 else "Fake News"
        
        return prediction, confidence, label

# Initialize the predictor
predictor = NewsPredictor()

@app.route('/')
def home():
    """Home page - Main analysis interface"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About Us page"""
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    """How It Works page"""
    return render_template('how_it_works.html')

@app.route('/contact')
def contact():
    """Contact Us page"""
    return render_template('contact.html')

@app.route('/documentation')
def documentation():
    """API Documentation page"""
    return render_template('documentation.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction route - handles news text analysis
    
    Accepts POST requests with news text and returns prediction
    """
    try:
        # Get the news text from the form
        news_text = request.form.get('news_text', '').strip()
        
        # Validate input
        if not news_text:
            return jsonify({
                'error': True,
                'message': 'Please enter some news text to analyze.'
            })
        
        if len(news_text) < 10:
            return jsonify({
                'error': True,
                'message': 'Please enter at least 10 characters for accurate analysis.'
            })
        
        # Make prediction
        prediction, confidence, label = predictor.predict(news_text)
        
        if prediction is None:
            return jsonify({
                'error': True,
                'message': 'Unable to analyze the text. Please try again.'
            })
        
        # Prepare response
        response = {
            'error': False,
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence),
            'confidence_percentage': round(confidence * 100, 2),
            'message': f"Analysis complete! This appears to be {label.lower()}."
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': True,
            'message': 'An error occurred during analysis. Please try again.'
        })

@app.route('/health')
def health():
    """
    Health check route - verifies if the model is loaded
    """
    model_status = "loaded" if predictor.model and predictor.vectorizer else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors
    """
    return jsonify({
        'error': True,
        'message': 'Internal server error. Please try again later.'
    }), 500

if __name__ == '__main__':
    """
    Run the Flask application
    """
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting AI-Based Fake News Detection Web Application")
    print("=" * 55)
    print(f"Server running on port: {port}")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Check if model is loaded
    if predictor.model and predictor.vectorizer:
        print("Model loaded successfully - Ready for predictions!")
    else:
        print("Warning: Model not loaded. Please run 'python model/train_model.py' first.")
    
    print("=" * 55)
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)