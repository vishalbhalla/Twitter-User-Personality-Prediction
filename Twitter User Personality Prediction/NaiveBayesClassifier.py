
from PreprocessTweets import PreprocessTweets
from FilterStopWords import FilterStopWords
from FeatureEngineering import FeatureEngineering
import nltk


def __init__(self):
        self.name = 'NaiveBayesClassifier'



objFilterStopWords = FilterStopWords()
objPreprocessTweets = PreprocessTweets()
objFeatureEngineering = FeatureEngineering()

trainingSet = objFeatureEngineering.createTrainingSet()

stopWordListFileName = 'TwitterData/StopWords.txt'
stopWords = objFilterStopWords.getStopWordList(stopWordListFileName)

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(trainingSet)

# Test the classifier
testTweet = 'Hurray, I am working on a project on personality prediction on twitter data using sentiment analysis!'
processedTestTweet = objPreprocessTweets.processTweet(testTweet)
featureVector = objFilterStopWords.getFeatureVector(processedTestTweet, stopWords)
print NBClassifier.classify(objFeatureEngineering.extract_features(featureVector))


# print informative features about the classifier
print NBClassifier.show_most_informative_features(10)


testTweet = 'I have successfully completed this project.'
processedTestTweet = objPreprocessTweets.processTweet(testTweet)
featureVector = objFilterStopWords.getFeatureVector(processedTestTweet, stopWords)
print NBClassifier.classify(objFeatureEngineering.extract_features(featureVector))
