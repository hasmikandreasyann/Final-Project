# Final-Project
## Data Science Part Report

### Introduction
Binary sentiment classification (Basic NLP)
This project is a graduate work to demonstrating my ability to solve the DS task and use MLE skills. 

### Exploratory Data Analysis (EDA)

I am loading the data from CSV files (train.csv and test.csv) and checking for missing values
Also visualizing the class distribution (sentiment) using a count plot and analyzing the distribution of review lengths, which provides insights into the dataset's characteristics.

### Feature Engineering
Tokenization: tokenizing the reviews using NLTK's word_tokenize function.
Stop-words Filtering: removing stop words using NLTK's English stop-word list.
Stemming vs. Lemmatization: comparing the effects of stemming and lemmatization on the tokenized reviews.
Vectorization: vectorizing the preprocessed text data using both Count Vectorization and TF-IDF Vectorization techniques.

### Modeling
I am training and evaluating several machine learning models, including Logistic Regression, Naive Bayes, and Random Forest, using TF-IDF vectorized data. Also providing the accuracy and classification reports for each model.

### Business Applications

#### Customer Feedback Analysis: 
Sentiment analysis can help businesses analyze customer feedback to identify areas for improvement and prioritize action items.

#### Brand Monitoring and Reputation Management:
It enables businesses to monitor brand reputation, identify PR crises, and maintain a positive brand image.

#### Market Research and Competitor Analysis: 
Sentiment analysis provides insights into consumer preferences, trends, and sentiments, aiding in market research and competitor analysis.

#### Product Development and Innovation:
By analyzing sentiment towards existing products, businesses can inform product development efforts and innovate new offerings.

#### Customer Service Optimization: 
Sentiment analysis can complement customer service operations by categorizing and prioritizing customer inquiries or complaints based on sentiment.

#### Risk Management:
It helps in monitoring market sentiment, identifying potential risks or opportunities, and making data-driven decisions to mitigate risks.
Brand Campaign Evaluation: Sentiment analysis evaluates the effectiveness of marketing campaigns by measuring changes in sentiment before, during, and after campaign launches.

#### Value for Business
Sentiment analysis provides businesses with actionable insights, including:
Understanding customer perceptions, preferences, and behaviors.
Improving customer experiences and satisfaction.
Making informed decisions to drive business growth.
Optimizing marketing and advertising strategies.
Enhancing brand reputation and loyalty.

#### Conclusion
In conclusion, sentiment analysis is a valuable tool for businesses to gain insights into customer sentiment, preferences, and behaviors. By leveraging sentiment analysis, businesses can make informed decisions, improve customer experiences, and drive business success.


## Machine Learning Part

### Modeling
Based on DS part
A logistic regression model is chosen as the machine learning algorithm for sentiment analysis. Logistic regression is well-suited for binary classification tasks like sentiment analysis and is known for its simplicity and interpretability. The model is initialized with default parameters and trained on the TF-IDF features and corresponding sentiment labels.

### Training
The training process begins by loading the training dataset (train.csv) using a custom function load_dataset. The dataset contains text reviews and their corresponding sentiment labels. Text preprocessing techniques are then applied to clean and prepare the data for modeling. This includes tokenization, removal of stopwords, and lemmatization to normalize the text data.
After preprocessing, the text data is converted into numerical features using TF-IDF vectorization. The TF-IDF vectorizer is trained on the lemmatized text data to transform it into a sparse matrix of TF-IDF features. These features represent the importance of each word in the text reviews relative to the entire dataset.

### Evaluation
Using the TF-IDF features of the test data, the logistic regression model makes predictions on the sentiment of the text reviews. This step involves applying the trained model to classify each review as either positive or negative sentiment.
The performance of the model is evaluated using two key metrics: accuracy and classification report. Accuracy measures the proportion of correctly classified reviews out of all reviews in the test dataset. The classification report provides a detailed breakdown of precision, recall, F1-score, and support for each sentiment category.

## How to Run Dockerfiles

###Clone the repository
git clone https://github.com/hasmikandreasyann/Final-Project/blob/main/src/train/Dockerfile

### Navigate to Cloned Repository
cd Final-Project

### Built a Docker image in train/inference
docker build -t train-image -f src/train/Dockerfile .
docker build -t inference-image -f src/inference/Dockerfile .

###Run a container
docker run train-image
docker run inference-image

### Troubleshooting
I do have a problem with inference/Dockerfile as it can not find the path but I
used the same method as in train/Dockerfile. I will try to correct that some way.

## Best Test Metric
The best test metric achieved during model evaluation is 0.8971.
