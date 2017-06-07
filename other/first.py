from nltk.book import *
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize, TreebankWordTokenizer

para = "Hello World. It's good to see you. Thanks for buying this book.\n"

x = sent_tokenize(para)
print(x)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
y = tokenizer.tokenize(para)
print(y)

a = word_tokenize(para)
print(a)

tokenizer = TreebankWordTokenizer()
b = tokenizer.tokenize(para)
print(b)
