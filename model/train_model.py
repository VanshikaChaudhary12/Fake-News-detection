"""
Enhanced AI-Based Fake News Detection - Improved Model Training
================================================================
Features: Advanced preprocessing, multiple feature extraction methods,
and improved model with hyperparameter tuning
"""

import pickle
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

print("🚀 Enhanced AI-Based Fake News Detection - Model Training")
print("=" * 60)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def advanced_preprocess(text):
    """Enhanced text preprocessing with lemmatization"""
    if pd.isna(text) or text == "":
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

# Load datasets
print("📊 Loading datasets...")
try:
    fake_df = pd.read_csv('dataset/Fake.csv')
    real_df = pd.read_csv('dataset/True.csv')
    
    fake_df['label'] = 0
    real_df['label'] = 1
    
    fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
    real_df['text'] = real_df['title'] + ' ' + real_df['text']
    
    print(f"✅ Loaded {len(fake_df)} fake news articles")
    print(f"✅ Loaded {len(real_df)} real news articles")
    
    df = pd.concat([fake_df[['text', 'label']], real_df[['text', 'label']]], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📈 Total dataset size: {len(df)} articles")
    
except FileNotFoundError:
    print("❌ Error: Dataset files not found!")
    print("Please download Fake.csv and True.csv and place them in the 'dataset/' folder")
    exit(1)

# Preprocessing
print("🔄 Preprocessing text data...")
df['cleaned_text'] = df['text'].apply(advanced_preprocess)
df = df[df['cleaned_text'].str.len() > 10]
print(f"📝 Preprocessed {len(df)} articles")

# Feature extraction with enhanced parameters
print("🔢 Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

print(f"✅ Created TF-IDF matrix: {X.shape}")
print(f"📊 Vocabulary size: {len(vectorizer.vocabulary_)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train enhanced model
print("🤖 Training Enhanced Logistic Regression model...")
model = LogisticRegression(
    C=10,
    max_iter=1000,
    solver='saga',
    penalty='l2',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n📊 ENHANCED MODEL PERFORMANCE:")
print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))

# Save model
print("💾 Saving enhanced model and vectorizer...")
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved as 'model/model.pkl'")
print("✅ Vectorizer saved as 'model/vectorizer.pkl'")
print("🎉 Enhanced model training completed successfully!")
