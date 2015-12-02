
# import PreprocessTweets
# import FilterStopWords
from PreprocessTweets import PreprocessTweets
from FilterStopWords import FilterStopWords

#Read the tweets one by one and process it
fp = open('TwitterData/UserTweets.txt', 'r')
line = fp.readline()

objFilterStopWords = FilterStopWords()
objPreprocessTweets = PreprocessTweets()

st = open('TwitterData/StopWords.txt', 'r')
stopWords = objFilterStopWords.getStopWordList('TwitterData/StopWords.txt')

while line:
    processedTweet = objPreprocessTweets.processTweet(line)
    featureVector = objFilterStopWords.getFeatureVector(processedTweet, stopWords)
    print featureVector
    line = fp.readline()
#end loop
fp.close()
