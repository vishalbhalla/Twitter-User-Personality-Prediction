import csv
from nltk.corpus import stopwords
from textblob.blob import TextBlob

from mmds.supervised.filter_stop_words import FilterStopWords
from mmds.supervised.preprocess_tweets import PreprocessTweets


class FeatureEngineering:

    def __init__(self):
        self.name = 'FeatureEngineering'
        self.featureList = []
        # self.sid = SentimentIntensityAnalyzer()


    # start extract_features
    def extract_features(self, tweet):
        tweet_words = set(tweet)
        features = {}
        for word in self.featureList:
            features['contains(%s)' % word] = (word in tweet_words)
        return features

# # Create New Training set based on personality labels predicted from Survey results

    def createNewTrainingSet(self, training_data_file):
        XTrain = []
        YTrain = []
        XTrainFeatures = []
        XTrainSentiment = []
        XTrainFreqTweets = []
        geo_latitude = []
        geo_longitude = []
        
        objFilterStopWords = FilterStopWords()
        objPreprocessTweets = PreprocessTweets()

        stopWords = objFilterStopWords.getStopWordList('../../TwitterData/StopWords.txt')
        
        # Read the tweets one by one and process it
        inpTweets = csv.reader(open(training_data_file, 'rb'), delimiter=',')
        inpTweets.next()
        tweets = []
        i = 0
        for row in inpTweets:
#             print row
            personality = row[5]
            tweet = row[1]
            cleanTweet = tweet.replace('"",""', " ")
            cleanTweet = cleanTweet.replace('""', " ")
            processedTweet = objPreprocessTweets.processTweet(cleanTweet)

            XTrainFreqTweets.append(int(row[4]))
            wordsList = processedTweet.split()
            
            # Remove stop words
            filtered_words = [word for word in wordsList if word not in stopwords.words('english')]
            filteredTweets = ' '.join(filtered_words)
            
            featureVector = objFilterStopWords.getFeatureVector(processedTweet, stopWords)
            
            geo_latitude.append(float(row[2]))
            geo_longitude.append(float(row[3]))
            
            blob = TextBlob(processedTweet)
            sentiment = 0
            for sentence in blob.sentences:
                sentiment += sentence.sentiment.polarity

            totSentiment = sentiment / len(blob.sentences)

            XTrainSentiment.append(totSentiment)

            XTrainFeatures.append(filteredTweets)
            
            YTrain.append(personality.replace('[', '').replace('\"', '').replace(']', ''))
                        
#             i+=1
#             if i==3:
#                 break
            

        return XTrain, YTrain, XTrainFeatures, XTrainSentiment, XTrainFreqTweets, geo_latitude, geo_longitude
