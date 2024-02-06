# src/feature_engineering.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib

def count_vectorization(text_data):
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(text_data.apply(lambda tokens: ' '.join(tokens)))
    return X_count

def tfidf_vectorization(text_data):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(text_data.apply(lambda tokens: ' '.join(tokens)))
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    return X_tfidf

import joblib

def load_tfidf_vectorizer():
    # Print a message indicating the start of the loading process
    print("Loading TF-IDF vectorizer...")

    try:
        # Attempt to load the TF-IDF vectorizer instance
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("TF-IDF vectorizer loaded successfully.")
        return tfidf_vectorizer
    except FileNotFoundError:
        # Handle the case where the file is not found
        print("Error: TF-IDF vectorizer file not found.")
        return None
    except Exception as e:
        # Handle any other exceptions that may occur during loading
        print(f"Error occurred while loading TF-IDF vectorizer: {str(e)}")
        return None



if __name__ == "__main__":


    # Example of TF-IDF vectorization
    #tfidf_vectorized_data = tfidf_vectorization(example_texts)
    print("\nTF-IDF Vectorized Data:")

