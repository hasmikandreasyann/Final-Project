# src/feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def count_vectorization(text_data):
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(text_data.apply(lambda tokens: ' '.join(tokens)))
    return X_count

def tfidf_vectorization(text_data):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(text_data.apply(lambda tokens: ' '.join(tokens)))
    return X_tfidf


if __name__ == "__main__":


    # Example of TF-IDF vectorization
    #tfidf_vectorized_data = tfidf_vectorization(example_texts)
    print("\nTF-IDF Vectorized Data:")

