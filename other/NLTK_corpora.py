from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

text = gutenberg.raw("bible-kjv.txt")

tokens = sent_tokenize(text)
for i in range(5):
    print(tokens[i])