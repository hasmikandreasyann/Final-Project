# run_inference.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = joblib.load('models/sentiment_model.pkl')  # Adjust based on your actual path

# Load test data
test_data = pd.read_csv('test.csv')  # Adjust based on your actual path

# Assuming you have 'filtered_reviews' or 'lemmatized_reviews' as your processed data
X_test = test_data['lemmatized_reviews']  # Change to 'filtered_reviews' if needed
y_test = test_data['sentiment']

# Vectorize using TF-IDF Vectorization (using the same vectorizer as in train.py)
vectorizer = TfidfVectorizer()
X_test_vectorized = vectorizer.transform(X_test.apply(lambda tokens: ' '.join(tokens)))

# Make predictions on the test set
predictions_test = model.predict(X_test_vectorized)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, predictions_test)
classification_rep = classification_report(y_test, predictions_test)

# Print or log the evaluation results
print(f"Test Accuracy: {accuracy}")
print("Test Classification Report:\n", classification_rep)

# Optionally, store the test metric in a file or use it as needed
with open('README.md', 'a') as readme:
    readme.write(f"\n\n## Best Test Metric\nThe best test metric achieved during model evaluation is {accuracy}.")