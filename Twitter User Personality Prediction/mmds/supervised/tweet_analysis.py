
# import preprocess_tweets
# import filter_stop_words
from mmds.supervised.filter_stop_words import FilterStopWords
from mmds.supervised.preprocess_tweets import PreprocessTweets

#Read the tweets one by one and process it
fp = open('../../TwitterData/UserTweets.txt', 'r')
line = fp.readline()

objFilterStopWords = FilterStopWords()
objPreprocessTweets = PreprocessTweets()

st = open('../../TwitterData/StopWords.txt', 'r')
stopWords = objFilterStopWords.getStopWordList('../../TwitterData/StopWords.txt')

while line:
    processedTweet = objPreprocessTweets.processTweet(line)
    featureVector = objFilterStopWords.getFeatureVector(processedTweet, stopWords)
    print featureVector
    line = fp.readline()
#end loop
fp.close()
