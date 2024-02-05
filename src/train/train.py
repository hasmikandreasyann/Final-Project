import nltk
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK punkt resource
nltk.download('punkt')
nltk.download('stopwords')

# Load data - adjust the path based on your directory structure
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent.parent / '/workspaces/Final-Project/data'  # Adjust based on your actual directory structure
train_data = pd.read_csv(data_dir / 'raw' / 'train.csv')

# Tokenization
train_data['tokenized_reviews'] = train_data['review'].apply(word_tokenize)

# Stop-words Filtering
stop_words = set(stopwords.words('english'))
train_data['filtered_reviews'] = train_data['tokenized_reviews'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

# Assuming you have 'filtered_reviews' or 'lemmatized_reviews' as your processed data
X = train_data['filtered_reviews']  # Change to 'lemmatized_reviews' if needed
y = train_data['sentiment']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize using TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train.apply(lambda tokens: ' '.join(tokens)))
X_val_vectorized = vectorizer.transform(X_val.apply(lambda tokens: ' '.join(tokens)))

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Make predictions on the validation set
predictions_val = nb_model.predict(X_val_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_val, predictions_val)
classification_rep = classification_report(y_val, predictions_val)

# Print or log the evaluation results
print(f"Validation Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

# Save the trained model
output_dir = current_dir / '../../outputs/models'  # Adjust based on your actual directory structure
output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
model_filename = output_dir / 'sentiment_model.pkl'
joblib.dump(nb_model, model_filename)
