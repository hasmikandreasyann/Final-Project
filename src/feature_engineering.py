# src/feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def count_vectorization(text_data):
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(text_data)
    return pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names_out())

def tfidf_vectorization(text_data):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(text_data)
    return pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

if __name__ == "__main__":
    # Example usage
    example_texts = ["This is the first document.", "This document is the second document.", "And this is the third one."]

    # Example of count vectorization
    count_vectorized_data = count_vectorization(example_texts)
    print("Count Vectorized Data:")
    print(count_vectorized_data)

    # Example of TF-IDF vectorization
    tfidf_vectorized_data = tfidf_vectorization(example_texts)
    print("\nTF-IDF Vectorized Data:")
    print(tfidf_vectorized_data)
