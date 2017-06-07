
"""
Created on Thu Feb  2 10:21:32 2017

@author: Gautham
"""

import custom_sentimentanalyzer as sent
text = input("Enter Text: ")

category, confidence = sent.sentiment(text)
print ("Text: ", text)
print ("Category: ", category)
print ("Confidence: {:0.2f}" .format(confidence))