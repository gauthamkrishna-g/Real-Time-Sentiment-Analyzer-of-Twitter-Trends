# Real-Time-Sentiment-Analyzer-of-Twitter-Trends

✔ The live Twitter data correlating to a given query from the user is classified into positive or negative. This real-time  
data is graphed, thereby giving a Trend Analysis of the given query overtime.

✔ The following two classifiers were created:

1. An ensemble of Sentiment Analysis classifiers ([custom_sentimentanalyzer.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/custom_sentimentanalyzer.py)), resulting in a voted  
majority classifier trained using nltk and scikit-learn is created and pickled, returning the category (>pos or neg)  
and the confidence factor (sentiment polarity).

2. Another [TextBlob](https://textblob.readthedocs.io/en/dev/) Sentiment Analysis classifier is used for returning the category and sentiment polarity.

✔ When the user enters a query, the Twitter data relevant to the query is streamed using [Tweepy](http://www.tweepy.org/), which is then  
parsed and cleaned to remove links and special characters. The data is then passed into both the classifiers to  
get the category and the confidence factor ([twittersentimentanalysis.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/twittersentimentanalysis.py)). The categories for both classifiers are  
then written into separate files.

✔ The [twitter-feed-custom.txt](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/twitter-feed-custom.txt) and 
[twitter-feed-textblob.txt](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/twitter-feed-textblob.txt) are then used for graphing the live data on every hit  
of the query while streaming, using matplotlib. The overall percentage of "pos" and "neg" is also updated for a  
realistic trend of the query ([custom_livetwittergraph.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/custom_livetwittergraph.py)) and ([textblob_livetwittergraph.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/textblob_livetwittergraph.py)).

✔ The existing pickles can be used for training, or you can train your own customized ensemble classifier by  
running [custom_train.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/custom_train.py).

✔ To test the sentiment of any text of your choice, [test_sentiment.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/test_sentiment.py) can be used.

✔ An example of ensemble classifier for custom sentiment analysis is done for Movie Reviews and can be \
found here ([ensemble_classifer.py](https://github.com/gauthkris/Real-Time-Sentiment-Analyzer-of-Twitter-Trends/blob/master/ensemble_classifier.py)).
