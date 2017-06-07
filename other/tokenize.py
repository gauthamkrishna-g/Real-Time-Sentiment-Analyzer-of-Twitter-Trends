from nltk.tokenize import sent_tokenize, word_tokenize
#ext = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
#text = 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'
with open("short_reviews/positive.txt", "r") as f:
    short_pos = f.readlines()
short_pos = [x.strip() for x in short_pos]    
with open("short_reviews/negative.txt", "r") as f:
    short_neg = f.readlines()
short_neg = [x.strip() for x in short_neg]

#print(sent_tokenize(text))
print(word_tokenize(short_pos[0]))