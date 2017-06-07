# -*- coding: utf-8 -*-

"""
Created on Tue Feb  7 15:22:56 2017

@author: Gautham
"""

import os
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import StreamListener
import re
import json
from textblob import TextBlob
import custom_sentimentanalyzer as sent
from keys_access_tokens import consumer_key, consumer_secret, access_token, access_secret
# Save your own keys and access tokens in "keys_access_tokens.py" acquired from Twitter Apps

out1 = "twitter-feed.txt"
os.remove(out1) if os.path.exists(out1) else None

out2 = "twitter-feed-textblob.txt"
os.remove(out2) if os.path.exists(out2) else None

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
class listener(StreamListener):
        
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = clean_tweet(all_data["text"])
        print (tweet)
        
        category, confidence = sent.sentiment(tweet)  # Custom-sentiment
        print ("Custom Sentiment : ", category, "-->", confidence)
        
        if confidence * 100 >= 80:
            output = open(out1, "a")
            output.write(category)
            output.write("\n")
            output.close()
        
        blob = TextBlob(tweet)
        confidence = blob.sentiment.polarity  # Textblob-sentiment
        if confidence > 0:
            category = "pos"
        else:
            category = "neg"            
        
        print ("TextBlob Sentiment : ", category, "-->", "{:0.2f}" .format(confidence))
        print ()
        
        output = open(out2, "a")
        output.write(category)
        output.write("\n")
        output.close()
            
        return True
    
    def on_error(self, status):
        print (status)
        
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

query = input("Enter your keyword to be searched in Twitter: ")

output = open(out1, "a")
output.write(query)
output.write("\n")
output.close()

output = open(out2, "a")
output.write(query)
output.write("\n")
output.close()

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[query])
