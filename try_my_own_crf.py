# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:56:38 2020

Largely inspired by 
https://www.analyticsvidhya.com/blog/2018/08/nlp-guide-conditional-random-fields-text-classification/
https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
https://github.com/AiswaryaSrinivas/DataScienceWithPython/blob/master/CRF%20POS%20Tagging.ipynb

@author: sbuer
"""

# data handling
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers

# NLP
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer

# Misc
from collections import Counter
import time
import random

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Use the tagged NYT ingredients dataset
data = pd.read_csv(r'd:\"data science"\nutrition\scripts\ingredient-phrase-tagger\nyt-ingredients-snapshot-2015.csv', 
				   encoding="utf-8", index_col=None)
data.tail(10)

words = list(set(data["Word"].values))
n_words = len(words)
print(n_words)

# Tags: 'name', 'qty', 'range_end', 'unit', 'comment'
# I will add the additional tag 'other', for parts of the input that do not
# appear in any of the tagged output

# I have to make sure to convert the quantities in the input (strings) to 
# floats to match the output (I can then convert back to strings)
# e.g. 1/4 --> 0.25
# Consider this function from ingredient-phrase-tagger:
def _parseNumbers(s):
    """
    Parses a string that represents a number into a decimal data type so that
    we can match the quantity field in the db with the quantity that appears
    in the display name. Rounds the result to 2 places.
    """
    ss = utils.unclump(s)

    m3 = re.match('^\d+$', ss)
    if m3 is not None:
        return decimal.Decimal(round(float(ss), 2))

    m1 = re.match(r'(\d+)\s+(\d)/(\d)', ss)
    if m1 is not None:
        num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
        return decimal.Decimal(str(round(num, 2)))

    m2 = re.match(r'^(\d)/(\d)$', ss)
    if m2 is not None:
        num = float(m2.group(1)) / float(m2.group(2))
        return decimal.Decimal(str(round(num, 2)))

    return None

# Try with one row
sentence = []

# Input words
input_words = WordPunctTokenizer().tokenize(row.input)

# Words in specific Tag
names = str(row.name)

qties = str(row.qty).strip().split()
range_ends = str(row.range_end).strip().split()
comments = str(row.comment).strip().split()

for word in input_words:
	print(word)
	if word in names:
		sentence.append((word, 'NAME'))
	elif word in qties:
		sentence.append((word, 'QTY'))
	elif word in range_ends:
		sentence.append((word, 'RANGE_END'))
	elif word in comments:
		sentence.append((word, 'COMMENT'))
	else:
		sentence.append((word, 'OTHER'))
		

doc = []
for i, row in enumerate(data.itertuples()):
	
	sentence = []
	for word in input_words:
		
	


for i, row in enumerate(data.itertuples()):
	print(i, row)
	if i == 2:
		break


def formatData(input_col, tag_cols):
	'''
	DESCRIPTION:
		Transform Ingredient DataFrame into list of lists of tuples, with each
		tuple denoting a word and its tag and each inner list denoting a 
		sentence.
	INPUT:
		input_col (str): Name of sentence column
		tag_cols (str): Column
	'''
		

# Check how it looks for one sentence
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)


# Get all sentences with Tags
sentences = getter.sentences


# Define features
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# Craft features
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


# Fit the CRF
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)


from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)


crf.fit(X, y)


# Inspect the model
import eli5

eli5.show_weights(crf, top=30)


# Improve CRF with regularization
crf = CRF(algorithm='lbfgs',
c1=10,
c2=0.1,
max_iterations=100,
all_possible_transitions=False)

pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y)
print(report)

crf.fit(X, y)


eli5.show_weights(crf, top=30)

















# Generate features. These are the default features that NER algorithm uses in 
# nltk. One can modify it for customization.
data = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]    
    tagged = nltk.pos_tag(tokens)    
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])


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















