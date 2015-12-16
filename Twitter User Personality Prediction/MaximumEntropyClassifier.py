
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

# Train the classifier (Maximum Entropy Classifier)
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(trainingSet, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)

# Test the classifier
testTweet = 'Hurray, I am working on a project on personality prediction on twitter data using sentiment analysis!'
processedTestTweet = objPreprocessTweets.processTweet(testTweet)
featureVector = objFilterStopWords.getFeatureVector(processedTestTweet, stopWords)
print MaxEntClassifier.classify(objFeatureEngineering.extract_features(featureVector))


# print informative features about the classifier
print MaxEntClassifier.show_most_informative_features(10)
