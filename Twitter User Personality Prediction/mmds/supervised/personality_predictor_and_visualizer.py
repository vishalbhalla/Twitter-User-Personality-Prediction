import logging
from mpl_toolkits.basemap import Basemap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from mmds.supervised.feature_engineering import FeatureEngineering
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(filename="../../supervised.log", level=logging.DEBUG, format="%(asctime)-15s %(threadName)s  %(message)s")

PERSONALITY_LABELS = ['Conscientiousness', 'Extrovert', 'Agreeable', 'Empathetic', 'Novelty Seeking', 'Perfectionist', 'Rigid',
                      'Impulsive', 'Psychopath', 'Obsessive']

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']

def mapLabels(class_name):
    if class_name in PERSONALITY_LABELS:
        return PERSONALITY_LABELS.index(class_name)
    else:
        pass

def writePredictedLabelFile(YPred):
    f = open("../../TwitterData/Predictions.csv", "w")
    f.write("Id,Label" + "\n")
    for i in xrange(len(YPred)):
        f.write(str(i) + "," + str(int(YPred[i])) + "\n")
    f.close()

def reverseMapLabels(index):
    if index < len(PERSONALITY_LABELS):
        return PERSONALITY_LABELS[index]
    else:
        return None

def GeoPlot(geo_longitude, geo_latitude, labels):

    fig = plt.figure(figsize=(20, 10))
    
    raw_data = {'latitude': geo_latitude, 'longitude': geo_longitude}

    df = pd.DataFrame(raw_data, columns=['latitude', 'longitude'])
    
    totSampleLen = len(labels)
#     print totSampleLen
    colors = ['blue', 'beige', 'red', 'green', 'magenta', 'yellow', 'cyan', 'aquamarine', 'azure', 'darkkhaki']
    
    m = Basemap(projection='gall', lon_0=0, lat_0=0, resolution='i')
#     x1,y1=map(geo_longitude, geo_latitude)
    x1, y1 = m(df['longitude'].values, df['latitude'].values)


    m.drawmapboundary(fill_color='black')  # fill to edge
    m.drawcountries()
    m.fillcontinents(color='white', lake_color='black')
    
#     m.scatter(x1, y1, marker='D',color='m', s=2)
    for i in xrange(totSampleLen):
        for k in xrange(10):
            if labels[i] == k:
#                 print x1[i], y1[i]
#                 print colors[k]
#                 m.scatter(x1[i], y1[i], marker='D',color=colors[k], s=2)
                m.plot(x1[i], y1[i], 'ro', color=colors[k])  # 'ro', markersize=6)

    
    for k in xrange(10):
        m.scatter(0, 0, marker='D', color=colors[k], s=2, label=reverseMapLabels(k))
    
    plt.title("Geo-tagging Personality Types for Twitter Users")
    # Place a legend to the right of this smaller figure.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def reverseMapSentiments(index):
    if index < len(SENTIMENT_LABELS):
        return SENTIMENT_LABELS[index]
    else:
        return None

def GeoSentimentPlot(geo_longitude, geo_latitude, sentiments):

    fig = plt.figure(figsize=(20, 10))
    
    raw_data = {'latitude': geo_latitude,
            'longitude': geo_longitude}

    df = pd.DataFrame(raw_data, columns=['latitude', 'longitude'])

    
    totSampleLen = len(sentiments)
    colors = ['red', 'blue', 'green']
    
    negLimit = 0
    posLimit = 0
    
    m = Basemap(projection='gall', lon_0=0, lat_0=0, resolution='i')
    
    x1, y1 = m(df['longitude'].values, df['latitude'].values)

    m.drawmapboundary(fill_color='black')
    m.drawcountries()
    m.fillcontinents(color='white', lake_color='black')
    
    for i in xrange(totSampleLen):
#         print sentiments[i]
        if sentiments[i] < negLimit:
            m.plot(x1[i], y1[i], 'ro', color=colors[0])
        elif sentiments[i] >= negLimit and sentiments[i] <= posLimit:
            m.plot(x1[i], y1[i], 'ro', color=colors[1])
        elif sentiments[i] > posLimit:
            m.plot(x1[i], y1[i], 'ro', color=colors[2])
    
    
    for k in xrange(3):
        m.scatter(0, 0, marker='D', color=colors[k], s=2, label=reverseMapSentiments(k))
    
    plt.title("Geo-tagging Sentiments of Twitter Users")
    # Place a legend to the right of this smaller figure.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    """
    Main script starts here.
    """
    logging.info("Inside main...")
    training_data_file = '../../TwitterData/survey_dump_with_tweet_count'
    evauluation_data_file = '../../TwitterData/survey_dump_geo_gt_8_1'

    objFeatureEngineering = FeatureEngineering()
    XTrain, YTrain, XTrainFeatures, XTrainSentiment, XTrainFreqTweets, geo_latitude, \
    geo_longitude = objFeatureEngineering.createNewTrainingSet(training_data_file)
    
    XEval, YEval, XEvalFeatures, XEvalSentiment, XEvalFreqTweets, eval_geo_latitude, \
    eval_geo_longitude = objFeatureEngineering.createNewTrainingSet(evauluation_data_file)
    
    YTrain = map(mapLabels, YTrain)
    YEval = map(mapLabels, YEval)
    
    XTrain = np.array(XTrainFeatures)
    YTrain = np.array(YTrain)
    
    logging.info("Number of training vectors XTrain:{}, target variables YTrain:{}".format(len(XTrain), len(YTrain)))
        
    XEval = np.array(XEvalFeatures)
    YEval = np.array(YEval)
    
    logging.info("Number of evaluation vectors XEval:{}, target variables YEval:{}".format(len(XEval), len(YEval)))
    
    # ### Split Train and Test data
    
    TRAINING_DATA_SET_SIZE = 60
    XTrainSamples = XTrain[0:TRAINING_DATA_SET_SIZE]
    YTrainSamples = YTrain[0:TRAINING_DATA_SET_SIZE]
    
    XTestSamples = XTrain[TRAINING_DATA_SET_SIZE:]
    YTestSamples = YTrain[TRAINING_DATA_SET_SIZE:]
    
    logging.info("No. of training samples XTrainSamples:{}, test samples XTestSamples:{}".format(len(XTrainSamples), len(XTestSamples)))
    
    trainSentimentSamples = np.array(XTrainSentiment[0:TRAINING_DATA_SET_SIZE])
    testSentimentSamples = np.array(XTrainSentiment[TRAINING_DATA_SET_SIZE:])
    trainFreqTweetSamples = np.array(XTrainFreqTweets[0:TRAINING_DATA_SET_SIZE])
    testFreqTweetSamples = np.array(XTrainFreqTweets[TRAINING_DATA_SET_SIZE:])
    
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(np.array(XTrainFeatures + XEvalFeatures))
    
    logging.info("Total features in training and evalution data:{}".format(len(vectorizer.get_feature_names())))
    
    XTr = vectorizer.transform(XTrainSamples)
    trainBagVector = XTr.toarray()
    XTe = vectorizer.transform(XTestSamples)
    testBagVector = XTe.toarray()

    XEv = vectorizer.transform(XEval)
    evalBagVector = XEv.toarray()
    
    logging.info("Dimension of training bag:{}, test bag:{}, eval bag".format(trainBagVector.shape,
                                                                testBagVector.shape, evalBagVector.shape))
    
    # join word features + sentiment + tweet frequency for training samples ... 
    XTrainAllFeatures = np.column_stack((np.column_stack((trainBagVector, trainSentimentSamples)), trainFreqTweetSamples))
    
    # join word features + sentiment + tweet frequency for testing samples ...  
    XTestAllFeatures = np.column_stack((np.column_stack((testBagVector, testSentimentSamples)), testFreqTweetSamples))
    
    # join word features + sentiment + tweet frequency for evalution samples ... 
    XEvalAllFeatures = np.column_stack((np.column_stack((evalBagVector, XEvalSentiment)), XEvalFreqTweets))
    
    logging.info("Dim of all training samples:{}, test samples:{}, eval samples, ytrain :{}".format(XTrainAllFeatures.shape,
                                                        XTestAllFeatures.shape, XEvalAllFeatures.shape, YTrainSamples.shape))
    
    """K Nearest Neighbourhood"""
    params = {'neighbours':25}
    neigh = KNeighborsClassifier(n_neighbors=params['neighbours'])
    YPred = neigh.fit(XTrainAllFeatures, YTrainSamples).predict(XEvalAllFeatures)
    
    """Random Forest"""
    params = {'trees':150, 'criterion':'entropy', 'random_state':None}
    clf = RandomForestClassifier(n_estimators=params['trees'], criterion=params['criterion'], random_state=params['random_state'])
    clf.fit(XTrainAllFeatures, YTrainSamples)
    YPred = clf.predict(XEvalAllFeatures)
    
    """SVM"""
    params = {'kernel':'rbf'}
    YPred = svm.SVC(kernel=params['kernel']).fit(XTrainAllFeatures, YTrainSamples).predict(XEvalAllFeatures)
    
    GeoPlot(eval_geo_longitude, eval_geo_latitude, YPred)
    
    GeoSentimentPlot(np.array(eval_geo_longitude), np.array(eval_geo_latitude), XEvalSentiment)
    
