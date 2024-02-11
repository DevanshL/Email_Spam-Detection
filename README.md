# Email_Spam-Detection

This project implements a machine learning model to classify emails as either ham (non-spam) or spam. It utilizes natural language processing (NLP) techniques to preprocess the email data and a machine learning algorithm for classification.

## Dataset

The dataset used for training and testing the classifier consists of a collection of labeled emails, where each email is labeled as either ham or spam. The dataset is typically split into a training set and a testing set for model development and evaluation.

## Preprocessing

Before training the machine learning model, the emails undergo preprocessing steps to clean and tokenize the text data. Preprocessing steps may include:

- Lowercasing the text
- Tokenization
- Removing stop words
- Stemming or lemmatization
- Handling special characters and punctuation

## Feature Extraction

To train the classifier, we extract features from the preprocessed email text. A common approach is to use the bag-of-words model or more advanced techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) to represent the text data as numerical features.

## Machine Learning Model

We train a machine learning model on the extracted features to classify emails as ham or spam. Popular algorithm for text classification is Naive Bayes.
## Evaluation

The performance of the classifier is evaluated using metrics such as accuracy, precision, recall, and F1-score on a held-out test set. Additionally, we may analyze the confusion matrix and ROC curve to assess the model's performance.


## Files

- `app.py`: Python script containing the Streamlit web application code.
- `mnb.pkl`: Serialized trained Multinomial Naive Bayes classifier.
- `vectorizer.pkl`: Serialized TF-IDF vectorizer used for feature extraction.
- `sms_spam.ipynb`: Jupyter Notebook containing the data exploration, preprocessing, model training, and evaluation steps.
- `spam.csv`: Dataset containing SMS messages labeled as spam or ham.

## Contributing

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
