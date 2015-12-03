
import csv
from PreprocessTweets import PreprocessTweets
from FilterStopWords import FilterStopWords
import nltk

featureList = []

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
        featureList.append(word)
    # featureList.append([featureVector[i] for i in xrange(len(featureVector))])

    # Extract sentiment based on the tweet.
    sentiment = ''
    featureVector.append(sentiment)

    tweets.append((featureVector, personality));
#end loop
print tweets
print featureList

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)

print featureList
print training_set
