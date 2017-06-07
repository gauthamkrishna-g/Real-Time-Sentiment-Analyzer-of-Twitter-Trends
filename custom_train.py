
"""
Created on Mon Jan  30 14:47:29 2017

@author: Gautham
"""

import random
import pickle
from nltk.probability import FreqDist
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.tokenize import word_tokenize
from nltk import pos_tag

short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []
all_words = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    part_of_speech = pos_tag(word_tokenize(p))
    for w in part_of_speech:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for n in short_neg.split('\n'):
    documents.append((n, "neg"))
    part_of_speech = pos_tag(word_tokenize(n))
    for w in part_of_speech:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickles/documents.pickle", "wb") 
pickle.dump(documents, save_documents)
save_documents.close()

print ("Documents Pickled!")
            
all_words = FreqDist(all_words)

word_features = list(all_words.keys())[:6000]

save_word_features = open("pickles/word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

print ("Word_Features Pickled!")

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print (len(featuresets))

train_data = featuresets[:10000]
test_data = featuresets[10000:]

NB_classifier = NaiveBayesClassifier.train(train_data)
print ("Naive Bayes Classifier Accuracy : ", (accuracy(NB_classifier, test_data)) * 100)
#NB_classifier.show_most_informative_features(15)

save_classifier = open("pickles/Naive_Bayes.pickle", "wb") 
pickle.dump(NB_classifier, save_classifier)
save_classifier.close()

print ("Naive Bayes Pickled!")

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_data)
print ("MNB Classifier Accuracy : ", (accuracy(MNB_classifier, test_data)) * 100)

save_classifier = open("pickles/Multinomial_NB.pickle", "wb") 
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()   

print ("Multinomial NB Pickled!")

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_data)
print ("BernoulliNB Classifier Accuracy : ", (accuracy(BernoulliNB_classifier, test_data)) * 100)

save_classifier = open("pickles/Bernoulli_NB.pickle", "wb") 
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

print ("Bernoulli NB Pickled!")

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_data)
print ("LogisticRegression Classifier Accuracy : ", (accuracy(LogisticRegression_classifier, test_data)) * 100)

save_classifier = open("pickles/Logistic_Regression.pickle", "wb") 
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

print ("Logistic Regression Pickled!")

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train_data)
print ("SGD Classifier Accuracy : ", (accuracy(SGD_classifier, test_data)) * 100)

save_classifier = open("pickles/SGD_Classifier.pickle", "wb") 
pickle.dump(SGD_classifier, save_classifier)
save_classifier.close()

print ("SGD Classifier Pickled!")

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_data)
print ("LinearSVC Classifier Accuracy : ", (accuracy(LinearSVC_classifier, test_data)) * 100)

save_classifier = open("pickles/Linear_SVC.pickle", "wb") 
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()   

print ("Linear SVC Pickled!")

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train_data)
print ("NuSVC Classifier Accuracy : ", (accuracy(NuSVC_classifier, test_data)) * 100)

save_classifier = open("pickles/Nu_SVC.pickle", "wb") 
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

print ("NuSVC Pickled!")
