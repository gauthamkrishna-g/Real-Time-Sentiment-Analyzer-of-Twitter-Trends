
"""
Created on Wed Feb  1 17:32:51 2017

@author: Gautham
"""

import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self, features):
        votes = []
        for c in self._classifers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        confidence_factor = choice_votes / len(votes)

        return confidence_factor

documents_f = open("pickles/documents.pickle", "rb") 
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickles/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

file = open("pickles/Naive_Bayes.pickle", "rb") 
NB_classifier = pickle.load(file)
file.close()

file = open("pickles/Multinomial_NB.pickle", "rb") 
MNB_classifier = pickle.load(file)
file.close()

file = open("pickles/Bernoulli_NB.pickle", "rb") 
BernoulliNB_classifier = pickle.load(file)
file.close()

file = open("pickles/Logistic_Regression.pickle", "rb") 
LogisticRegression_classifier = pickle.load(file)
file.close()

file = open("pickles/SGD_Classifier.pickle", "rb") 
SGD_Classifier = pickle.load(file)
file.close()

file = open("pickles/Linear_SVC.pickle", "rb") 
LinearSVC_classifier = pickle.load(file)
file.close()

file = open("pickles/Nu_SVC.pickle", "rb") 
NuSVC_classifier = pickle.load(file)
file.close()

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BernoulliNB_classifier,
                                  LogisticRegression_classifier, LinearSVC_classifier)
                                  
def sentiment(text):
    features = find_features(text)
    return voted_classifier.classify(features), voted_classifier.confidence(features)
