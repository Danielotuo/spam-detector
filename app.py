# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:56:33 2020

@author: danid
"""

import re
import nltk
from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pickle
import streamlit as st

def main():
    '''
    load the machine learning models
    preprocess the input
    MUltinomialNB to predict the output
    Countvectorizer to create bag of words
    '''
    
    # load the model
    filename = 'model.pkl'
    clf = pickle.load(open(filename, 'rb'))
    cv=pickle.load(open('tranform.pkl','rb'))

    
    st.title('Spam Message detector')
    st.header('A simple Machine learning web app that uses nltk and Naives Bayes Classifier to predict spam messages')
    message = st.text_area('Message:', 'Type here..')
    msg = nltk.sent_tokenize(message) 
    stemmer = PorterStemmer() 
    corpus = []
    
    # Preprocess the message
    for i in range(len(msg)):
        words = re.sub('[^a-zA-Z]', ' ', msg[i])
        words = words.lower()
        words = words.split()
        words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
        words = ' '.join(words)
        corpus.append(words)
    
    # use the ML models to predict output
    vect = cv.transform(corpus).toarray()
    predictions = clf.predict(vect)
    
    st.write(predictions)
    
    if st.button('Predict'):
        
        with st.spinner('Analyzing the text …'):
            if predictions.any() == 1:
                st.success("spam")
            
            elif predictions.any() == 0:
                st.success("ham")
            
if __name__ == '__main__':
    main()





# df = pd.read_csv('spam_ham_dataset.csv')

# # drop redundant columns
# df = df.drop(df.columns[[0, 3]], axis=1)

# # Change the target variable(ham or spam) to dummy variables
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# X = df['text']
# y = df['label']

# # Data preprocessing and cleaning




# def preprocess(x):
#     stemmer = PorterStemmer()
#     corpus = []
#     # for each text, retain only letters and remove stopwords
#     for i in range(0, len(df)):
#         words = re.sub('[^a-zA-Z]', ' ', df['text'][i])
#         words = words.lower()
    
#         # remove the text subject from the message which is the first word
#         words = words.split()[1:]
    
#         words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
#         words = ' '.join(words)
#         corpus.append(words)
#     return corpus

# corpus_ = preprocess(df['text'])
    
# # Creating the bag of words model and taking the top 10000 frequent words
# from sklearn.feature_extraction.text import CountVectorizer

# cv = CountVectorizer()
# X = cv.fit_transform(corpus_).toarray()

# # Save the preprocessing model
# pickle.dump(cv, open('tranform.pkl', 'wb'))




# # Split into training and test
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# # Train model with Naive bayes classifier
# from sklearn.naive_bayes import MultinomialNB

# clf = MultinomialNB().fit(X_train, y_train)

# # predict
# y_pred = clf.predict(X_test)

# # clf.score(X_test, y_test)

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# from sklearn.metrics import accuracy_score

# accuracy = accuracy_score(y_test, y_pred)

# # Save the model

# filename = 'model.pkl'
# pickle.dump(clf, open(filename, 'wb'))
