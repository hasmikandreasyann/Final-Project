# src/train/train.py
# src/train/train.py

# train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_load import load_dataset
import pickle
from src.text_processing import tokenize_text, remove_stopwords, apply_lemmatization
from src.feature_engineering import tfidf_vectorization
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
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

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Build and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
model_filename = os.path.join(current_dir, '..', '..', 'outputs', 'models', 'logistic_regression_model.pkl')
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Load test dataset
test_data = load_dataset('test.csv')

# Text processing for test data
test_data['tokenized_reviews'] = test_data['review'].apply(tokenize_text)
test_data['filtered_reviews'] = test_data['tokenized_reviews'].apply(remove_stopwords)
test_data['lemmatized_reviews'] = test_data['filtered_reviews'].apply(apply_lemmatization)

