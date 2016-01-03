
import csv
from PreprocessTweets import PreprocessTweets
from FilterStopWords import FilterStopWords
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

class FeatureEngineering:

    def __init__(self):
        self.name = 'FeatureEngineering'
        self.featureList = []
        # self.sid = SentimentIntensityAnalyzer()


    #start extract_features
    def extract_features(self,tweet):
        tweet_words = set(tweet)
        features = {}
        for word in self.featureList:
            features['contains(%s)' % word] = (word in tweet_words)
        return features


    def createTrainingSet(self):

        objFilterStopWords = FilterStopWords()
        objPreprocessTweets = PreprocessTweets()

        stopWords = objFilterStopWords.getStopWordList('TwitterData/StopWords.txt')

        #Read the tweets one by one and process it
        inpTweets = csv.reader(open('TwitterData/labeledPersonalityTweets.csv', 'rb'), delimiter=',', quotechar='|')
        tweets = []
        for row in inpTweets:
            personality = row[0]
            tweet = row[1]
            processedTweet = objPreprocessTweets.processTweet(tweet)
            featureVector = objFilterStopWords.getFeatureVector(processedTweet, stopWords)

            # Append to feature list to collect total words
            for word in featureVector:
                self.featureList.append(word)
            # featureList.append([featureVector[i] for i in xrange(len(featureVector))])

            # Use NLTK Vader for Sentiment Analysis

            # Citation: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            # Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
            # Extract sentiment based on the tweet.
            # ss = self.sid.polarity_scores(row)
            # for k in sorted(ss):
            #     print('{0}: {1}, '.format(k, ss[k]))
            #
            # totSentiment = sorted(ss)[0]

            # Use TextBlog for Sentiment Analysis
            print tweet
            blob = TextBlob(tweet)
            print blob
            sentiment = 0
            for sentence in blob.sentences:
                print sentence
                sentiment += sentence.sentiment.polarity
                print sentiment

            totSentiment = sentiment/ len(blob.sentences)
            featureVector.append(totSentiment)

            tweets.append((featureVector, personality));
        #end loop
        print tweets
        print self.featureList
        # Remove featureList duplicates
        featureList = list(set(self.featureList))

        # Extract feature vector for all tweets in one shote
        training_set = nltk.classify.util.apply_features(self.extract_features, tweets)

        print self.featureList
        print training_set
        return training_set





