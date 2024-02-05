# text_processing.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def apply_stemming(tokens):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(word) for word in tokens]

def apply_lemmatization(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    return [wordnet_lemmatizer.lemmatize(word) for word in tokens]

if __name__ == "__main__":
    # Example usage
    example_text = "This is an example text for processing."
    
    tokens = tokenize_text(example_text)
    print("Tokens:", tokens)

    without_stopwords = remove_stopwords(tokens)
    print("Without Stopwords:", without_stopwords)

    stemmed_tokens = apply_stemming(without_stopwords)
    print("After Stemming:", stemmed_tokens)

    lemmatized_tokens = apply_lemmatization(without_stopwords)
    print("After Lemmatization:", lemmatized_tokens)
