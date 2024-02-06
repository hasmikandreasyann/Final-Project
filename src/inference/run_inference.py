import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import joblib
from src.data_load import load_dataset
from src.feature_engineering import tfidf_vectorization, load_tfidf_vectorizer
from src.text_processing import tokenize_text, remove_stopwords, apply_lemmatization
from src.evaluation import evaluate_model

if __name__ == "__main__":
    # Load the trained model
    model = joblib.load('outputs/models/logistic_regression_model2.pkl')  

    # Load test data
    test_data = load_dataset('test.csv')

    # Text processing for test data
    test_data['tokenized_reviews'] = test_data['review'].apply(tokenize_text)
    test_data['filtered_reviews'] = test_data['tokenized_reviews'].apply(remove_stopwords)
    test_data['lemmatized_reviews'] = test_data['filtered_reviews'].apply(apply_lemmatization)  
    
    y_test = test_data['sentiment']
    tfidf_vectorizer = load_tfidf_vectorizer()
    X_test_tfidf = tfidf_vectorizer.transform(test_data['lemmatized_reviews'].apply(lambda tokens: ' '.join(tokens)))


    # Make predictions on the test set
    predictions_test = model.predict(X_test_tfidf)

    # Evaluate the model on the test set using the imported evaluate_model function
    accuracy, classification_rep = evaluate_model(predictions_test, y_test)

    # Print the evaluation results
    print(f"Test Accuracy: {accuracy}")
    print("Test Classification Report:\n", classification_rep)

    # Optionally, store the test metric in a file or use it as needed
    with open('README.md', 'a') as readme:
        readme.write(f"\n\n## Best Test Metric\nThe best test metric achieved during model evaluation is {accuracy}.")
