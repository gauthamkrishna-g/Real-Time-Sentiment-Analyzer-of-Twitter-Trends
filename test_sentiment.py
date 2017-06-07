
"""
Created on Thu Feb  2 10:21:32 2017

@author: Gautham
"""

import custom_sentimentanalyzer as sent
from textblob import TextBlob

text = input("Enter Text: ")

print ("Text: ", text)
print ()
print ("Custom Sentiment: ")
category, confidence = sent.sentiment(text)
print ("Category: ", category)
print ("Confidence: {:0.2f}" .format(confidence))

print ()

print ("TextBlob Sentiment: ")
blob = TextBlob(text)
confidence = blob.sentiment.polarity
if confidence > 0:
    category = "pos"
else:
    category = "neg"
print ("Category: ", category)
print ("Confidence: {:0.2f}" .format(confidence))
