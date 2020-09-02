# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:57:15 2020

ML model to detect spam messages
@author: danid
"""

import re
from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import pickle


df = pd.read_csv('spam_ham_dataset.csv')

# drop redundant columns
df = df.drop(df.columns[[0, 3]], axis=1)

# Change the target variable(ham or spam) to dummy variables
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label']

# Data preprocessing and cleaning
def preprocess(x):
    stemmer = PorterStemmer()
    corpus = []
    # for each text, retain only letters and remove stopwords
    for i in range(0, len(df)):
        words = re.sub('[^a-zA-Z]', ' ', df['text'][i])
        words = words.lower()
    
        # remove the text subject from the message which is the first word
        words = words.split()[1:]
    
        words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
        words = ' '.join(words)
        corpus.append(words)
    return corpus

corpus_ = preprocess(df['text'])
    
# Creating the bag of words model and taking the top 10000 frequent words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(corpus_).toarray()

# Save the preprocessing model
pickle.dump(cv, open('tranform.pkl', 'wb'))

# Split into training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train model with Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

# Save the model
filename = 'model.pkl'
pickle.dump(clf, open(filename, 'wb'))

