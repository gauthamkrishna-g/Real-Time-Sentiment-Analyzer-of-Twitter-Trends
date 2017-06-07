from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
text1 = ["thatta", "thatting", "thattaed"]
for w in text1:
    print ps.stem(w)

text2 = "What ever I had done previously has been erased, because the erasal of the erased documents is not an erase until the erased text are being erased completely."
word_tokens = word_tokenize(text2)

for w in word_tokens:
    print ps.stem(w),

#print(word_tokens)
