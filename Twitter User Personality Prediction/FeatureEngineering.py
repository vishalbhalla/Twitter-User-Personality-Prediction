
import csv
from PreprocessTweets import PreprocessTweets
from FilterStopWords import FilterStopWords
import nltk

class FeatureEngineering:

    def __init__(self):
        self.name = 'FeatureEngineering'
        self.featureList = []

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

            # Extract sentiment based on the tweet.
            sentiment = ''
            featureVector.append(sentiment)

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





