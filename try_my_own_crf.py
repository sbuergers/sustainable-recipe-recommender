# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:56:38 2020

Largely inspired by 
https://www.analyticsvidhya.com/blog/2018/08/nlp-guide-conditional-random-fields-text-classification/

@author: sbuer
"""

#invoke libraries
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os, os.path, sys
import glob
from xml.etree import ElementTree
import numpy as np
from sklearn.metrics import classification_report



# Generate features. These are the default features that NER algorithm uses in 
# nltk. One can modify it for customization.
data = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]    
    tagged = nltk.pos_tag(tokens)    
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

# Common features for all words. You may add more features here based on your custom use case
features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

# Features for words that are not at the beginning of a document
if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

# Features for words that are not at the end of a document
if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

return features






# Now we’ll build features and create train and test data frames.
X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)







# Let’s test our model.
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]






# You can inspect any predicted value by selecting the corresponding row 
# number “i”.
i = 0
for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s)" % (y, x))






# Check the performance of the model.

# Create a mapping of labels to indices
labels = {"claim_number": 1, "claimant": 1,"NA": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])






# Print out the classification report. Based on the model performance, build 
# better features to improve the performance.
print(classification_report(
    truths, predictions,
    target_names=["claim_number", "claimant","NA"]))







# predict new data
with codecs.open("D:/ SampleEmail6.xml", "r", "utf-8") as infile:
    soup_test = bs(infile, "html5lib")

docs = []
sents = []

for d in soup_test.find_all("document"):
   for wrd in d.contents:    
    tags = []
    NoneType = type(None)   

    if isinstance(wrd.name, NoneType) == True:
        withoutpunct = remov_punct(wrd)
        temp = word_tokenize(withoutpunct)
        for token in temp:
            tags.append((token,'NA'))            
    else:
        withoutpunct = remov_punct(wrd)
        temp = word_tokenize(withoutpunct)
        for token in temp:
            tags.append((token,wrd.name))
    #docs.append(tags)

sents = sents + tags # puts all the sentences of a document in one element of the list
docs.append(sents) #appends all the individual documents into one list       

data_test = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]    
    tagged = nltk.pos_tag(tokens)    
    data_test.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])

data_test_feats = [extract_features(doc) for doc in data_test]
tagger.open('crf.model')
newdata_pred = [tagger.tag(xseq) for xseq in data_test_feats]

# Let's check predicted data
i = 0
for x, y in zip(newdata_pred[i], [x[1].split("=")[1] for x in data_test_feats[i]]):
    print("%s (%s)" % (y, x))











# eof















