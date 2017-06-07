import random
from nltk.corpus import  movie_reviews
from nltk.probability import FreqDist

documents = [(list(movie_reviews.words(fileid)), category) 
            for category in movie_reviews.categories() 
            for fileid in movie_reviews.fileids(category)]
                
random.shuffle(documents)                                
print(documents[1])

all_words = [w.lower() for w in movie_reviews.words()]
#print(len(all_words))
all_words = FreqDist(all_words)
#print(len(all_words))
#print(all_words.most_common(15))
#print(all_words["though"])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]           