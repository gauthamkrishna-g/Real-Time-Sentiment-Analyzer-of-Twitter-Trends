import random
from nltk.corpus import  movie_reviews
from nltk.probability import FreqDist

documents = [(list(movie_reviews.words(fileid)), category) 
            for category in movie_reviews.categories() 
            for fileid in movie_reviews.fileids(category)]
                
random.shuffle(documents)                                
#print(documents[1])

all_words1 = [w.lower() for w in movie_reviews.words()]
print (len(all_words1))
all_words = FreqDist(all_words1)

print (len(all_words))
print (all_words.most_common(15))
print (all_words["stupid"])