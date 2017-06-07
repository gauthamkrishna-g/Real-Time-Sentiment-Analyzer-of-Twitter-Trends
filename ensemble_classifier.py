import random
import pickle
from nltk.corpus import  movie_reviews
from nltk.probability import FreqDist
#from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

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

documents = [(list(movie_reviews.words(fileid)), category) 
            for category in movie_reviews.categories() 
            for fileid in movie_reviews.fileids(category)]
                
random.shuffle(documents)                                

all_words = [w.lower() for w in movie_reviews.words()]
all_words = FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

#positive
train = featuresets[:1900]
test = featuresets[1900:]

#negative bias
#train = featuresets[100:]
#test = featuresets[:100]

#classifier = NaiveBayesClassifier.train(train)
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

print ("Naive Bayes Classifier Accuracy : ", (accuracy(classifier, test)) * 100)

classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train)
print ("MNB Classifier Accuracy : ", (accuracy(MNB_classifier, test)) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train)
print ("BernoulliNB Classifier Accuracy : ", (accuracy(BernoulliNB_classifier, test)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train)
print ("LogisticRegression Classifier Accuracy : ", (accuracy(LogisticRegression_classifier, test)) * 100)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(train)
print ("SGD Classifier Accuracy : ", (accuracy(SGD_classifier, test)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train)
print ("LinearSVC Classifier Accuracy : ", (accuracy(LinearSVC_classifier, test)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(train)
print ("NuSVC Classifier Accuracy : ", (accuracy(NuSVC_classifier, test)) * 100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, 
                                  SGD_classifier, LinearSVC_classifier, NuSVC_classifier)
                                  
print ("Voted Classifier Accuracy : ", (accuracy(voted_classifier, test)) * 100)

#print (test[0][0])

print ("Classification : ", voted_classifier.classify(test[0][0]),
                                                     "\nConfidence Factor : ", voted_classifier.confidence(test[0][0])*100)                                                  
