# src/train/train.py
# src/train/train.py

# train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_load import load_dataset
from src.text_processing import tokenize_text, remove_stopwords, apply_lemmatization
from src.feature_engineering import tfidf_vectorization
import joblib
from sklearn.linear_model import LogisticRegression


# Load training dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
train_data = load_dataset('train.csv')

# Text processing for training data
train_data['tokenized_reviews'] = train_data['review'].apply(tokenize_text)
train_data['filtered_reviews'] = train_data['tokenized_reviews'].apply(remove_stopwords)
train_data['lemmatized_reviews'] = train_data['filtered_reviews'].apply(apply_lemmatization)

# Feature engineering for training data
X_train_tfidf = tfidf_vectorization(train_data['lemmatized_reviews'])
y_train = train_data['sentiment']

# Load the vectorizer instance whenever you need to transform new data
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Build and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Directory to save the model
output_dir = os.path.join(current_dir, '..', '..', 'outputs', 'models')
    
# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the trained model using joblib
model_filename = os.path.join(output_dir, 'logistic_regression_model2.pkl')
joblib.dump(model, model_filename)