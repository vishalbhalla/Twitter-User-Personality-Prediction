import logging
from mpl_toolkits.basemap import Basemap
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mmds.supervised.feature_engineering import FeatureEngineering


LOGGER = logging.getLogger("supervised")

def mapLabels(className):
    if className == 'Conscientiousness':
        return 0
    elif className == 'Extrovert':
        return 1
    elif className == 'Agreeable':
        return 2
    elif className == 'Empathetic':
        return 3
    elif className == 'Novelty Seeking':
        return 4
    elif className == 'Perfectionist':
        return 5
    elif className == 'Rigid':
        return 6
    elif className == 'Impulsive':
        return 7
    elif className == 'Psychopath':
        return 8
    elif className == 'Obsessive':
        return 9
#     elif className == None:
#         return 10
    else:
        pass

def createStateTransitionVector(categoricalState, stateDict, maxLength):
    if categoricalState:
        feature = []
        for state in categoricalState.split(' '):
            try:
                feature.append(stateDict[state.lower()])
            except KeyError:
                pass
#                 print state
        if len(feature) != maxLength:
            for i in xrange(maxLength - len(feature)):
                feature.append(0)
        assert(len(feature) == maxLength)
        return feature
    else:
        return [0] * maxLength


# In[36]:

def createStateVectors(XStates, stateDict, maxLength):
    XFeatures = []
    for state in XStates:
        XFeatures.append(createStateTransitionVector(state, stateDict, maxLength))
    return XFeatures

# ### Write Predicted Output Labels to File

# In[44]:

def writePredictedLabelFile(YPred):
    f = open("Predictions.csv", "w")
    f.write("Id,Label" + "\n")
    for i in xrange(len(YPred)):
        f.write(str(i) + "," + str(int(YPred[i])) + "\n")
    f.close()


# ### Classifiers

# Random Forest Classifier
# def classifyRandomForestClassifier(XTrain, XTest, YTrain, YTest,trees=100,crit='gini'):
def classifyRandomForestClassifier(XTrain, XTest, YTrain, YTest, params):
    trees = params['trees']
    crit = params['criterion']
    seed = params['random_state']
    clf = RandomForestClassifier(n_estimators=trees, criterion=crit, random_state=seed)
    clf.fit(XTrain, YTrain)
    YPred = clf.predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score) / (YPred.size)


# In[46]:

# Multi Class SVM
def classifyMultiClassSVMClassifier(XTrain, XTest, YTrain, YTest, params):
    ker = params['kernel']
    YPred = svm.SVC(kernel=ker).fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score) / (YPred.size)


# In[47]:

# K Nearest Neighbours Classifier
def classifyKNNClassifier(XTrain, XTest, YTrain, YTest, params):
#     print XTrain.shape, XTest.shape
    neighbours = params['neighbours']
    neigh = KNeighborsClassifier(n_neighbors=neighbours)
    YPred = neigh.fit(XTrain, YTrain).predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score) / (YPred.size)


# In[48]:

# Logistic Regression
def classifyLogisticRegression(XTrain, XTest, YTrain, YTest, params):
    LogReg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    LogReg.fit(XTrain, YTrain)
    # Finds the optimal model parameters using a least squares method.
    # To get the parameter values:
    # LogReg.get_params()
    # To predict a new input XTest,
    YPred = LogReg.predict(XTest)
    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score) / (YPred.size)


# In[49]:

# Adaboost Classfier
def classifyAdaboostClassifier(XTrain, XTest, YTrain, YTest, params):
    depth = params['max_depth']
    algo = params['algorithm']
    estimators = params['n_estimators']
    
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         algorithm=algo,
                         n_estimators=estimators)

    bdt.fit(XTrain, YTrain)
    YPred = bdt.predict(XTest)

    diff = YPred - YTest
    score = diff[diff == 0].size
    return (100.0 * score) / (YPred.size)


# In[50]:

# Neural Networks
try:
    from sknn.mlp import Classifier, Layer
except ImportError:
    print 'Please install scikit-neuralnetwork(pip install scikit-neuralnetwork)'

def classifyNeuralNetworkClassifier(XTrain, XTest, YTrain, YTest, params):
    activation = params['activation']
    actLastLayer = params['actLastLayer']
    rule = params['rule']
    noOfUnits = params['units']
    rate = params['rate']
    noOfIter = params['iter']
    nn = Classifier(layers=[Layer(activation, units=noOfUnits), Layer(actLastLayer)], learning_rule=rule,
        learning_rate=0.02,
        n_iter=10)
    nn.fit(XTrain, YTrain)
    YPred = nn.predict(XTest)
    diff = YPred - YTest.reshape(YPred.shape)
    score = diff[diff == 0].size
    score = (100.0 * score) / (YPred.size)
    return score


# ### Stratified K Fold Cross Validation

# In[51]:

def stratifiedKFoldVal(XTrain, YTrain, classify, params):
    n_folds = 5
    score = 0.0
    skf = StratifiedKFold(YTrain, n_folds)
    try:
        multi = params['multi']
    except KeyError:
        multi = False
    for train_index, test_index in skf:
        y_train, y_test = YTrain[train_index], YTrain[test_index]
        if not multi:
            X_train, X_test = XTrain[train_index], XTrain[test_index]
            score += classify(X_train, X_test, y_train, y_test, params)
        else:
            X_train, X_test = [XTrain[i] for i in train_index], [XTrain[i] for i in test_index]
            score += classify(np.array(X_train), np.array(X_test), y_train, y_test, params)
        
    return score / n_folds


# ### Normalisation of Feature Vectors

# In[52]:

def NormalizeVector(XTestFeatures, XTrainFeatures):
    XTestFeaturesNorm = preprocessing.normalize(XTestFeatures, norm='l2')
    XTrainFeaturesNorm = preprocessing.normalize(XTrainFeatures, norm='l2')
    print XTrainFeaturesNorm.shape, XTestFeaturesNorm.shape
#     print XTrainFeaturesNorm[0],XTestFeaturesNorm[0]
    return XTrainFeaturesNorm, XTestFeaturesNorm


def featNLTKClassify(samples, phase):
    featureVectors = vectorizer.get_feature_names()
    nltkClassifySamples = []

    for i in xrange(len(samples)):
        t = samples[i]
        lstFuncCalls = t.split()
        wordOccDict = {}
        for j in xrange(len(featureVectors)):
            wordOccDict[featureVectors[j]] = lstFuncCalls.count(featureVectors[j])
        if phase == 'train':
            nltkClassifySamples.append((wordOccDict, YTrain[i]))
        else:
            nltkClassifySamples.append(wordOccDict)

    return nltkClassifySamples


def reverseMapLabels(className):
    if className == 0:
        return 'Conscientiousness'
    elif className == 1:
        return 'Extrovert'
    elif className == 2:
        return 'Agreeable'
    elif className == 3:
        return 'Empathetic'
    elif className == 4:
        return 'Novelty Seeking'
    elif className == 5:
        return 'Perfectionist'
    elif className == 6:
        return 'Rigid'
    elif className == 7:
        return 'Impulsive'
    elif className == 8:
        return 'Psychopath'
    elif className == 9:
        return 'Obsessive'
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

def reverseMapSentiments(classNo):
    if classNo == 0:
        return 'Negative'
    elif classNo == 1:
        return 'Neutral'
    elif classNo == 2:
        return 'Positive'
    else:
        return None


# In[87]:

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
    LOGGER.info("Inside main...")
    objFeatureEngineering = FeatureEngineering()
    fileName = '../../TwitterData/survey_dump_with_tweet_count'
    XTrain, YTrain, XTrainFeatures, XTrainSentiment, XTrainFreqTweets, geo_latitude, geo_longitude \
 = objFeatureEngineering.createNewTrainingSet(fileName)
    
    fileName = '../../TwitterData/survey_dump_geo_gt_8_1'
    XEval, YEval, XEvalFeatures, XEvalSentiment, XEvalFreqTweets, eval_geo_latitude, \
    eval_geo_longitude = objFeatureEngineering.createNewTrainingSet(fileName)
    
    newYTrain = []
    # print YTrain
    for item in YTrain:
        temp = item.replace('[', '')
        temp = temp.replace('\"', '')
        newItem = temp.replace(']', '')
        newYTrain.append(newItem)
        
    YTrain = newYTrain
    
    YTrain = [mapLabels(x) for x in YTrain]
    YEval = [mapLabels(x) for x in YEval]
    
    
    print YEval[1:5]
    
    XTrain = np.array(XTrainFeatures)
    YTrain = np.array(YTrain)
    
    print len(XTrain)
    print len(YTrain)
    
    print XTrain[1]
    print YTrain[1]
    
    print XTrain[15]
    print YTrain[15]
    
    XEval = np.array(XEvalFeatures)
    YEval = np.array(YEval)
    
    
    # ### Split Train and Test data
    
    trainSamples = XTrain[0:60]
    YtrainSamples = YTrain[0:60]
    
    testSamples = XTrain[60:]
    YtestSamples = YTrain[60:]
    
    print len(trainSamples)
    print len(testSamples)
    
    # print XTrain[60:63]
    print len(XTrain)
    
    
    trainSentimentSamples = np.array(XTrainSentiment[0:60])
    testSentimentSamples = np.array(XTrainSentiment[60:])
    trainFreqTweetSamples = np.array(XTrainFreqTweets[0:60])
    testFreqTweetSamples = np.array(XTrainFreqTweets[60:])
    
    vectorizer = CountVectorizer()
    XTr = vectorizer.fit_transform(trainSamples)
    print len(vectorizer.get_feature_names())
    trainBagVector = XTr.toarray()
    print trainBagVector.shape
    XTe = vectorizer.transform(testSamples)
    testBagVector = XTe.toarray()
    print testBagVector.shape
    
    XEv = vectorizer.fit_transform(XEval)
    print len(vectorizer.get_feature_names())
    evalBagVector = XEv.toarray()
    print evalBagVector.shape
    
    evalBagVector = evalBagVector[:, 0:4914]
    print evalBagVector.shape
    
    stateDict = {}
    featureVectors = vectorizer.get_feature_names()
    for i in xrange(len(featureVectors)):
        stateDict[featureVectors[i]] = i + 1
    print len(stateDict), len(featureVectors)  # , stateDict
    
    trainStateTransitionVector = createStateVectors(trainSamples, stateDict, 9353)
    testStateTransitionVector = createStateVectors(testSamples, stateDict, 9353)
    # print trainStateTransitionVector[:2], testStateTransitionVector[:2]
    
    print max([len(i) for i in trainStateTransitionVector])
    print max([len(i) for i in testStateTransitionVector])
    
    noNGram = 3
    vectorizerNGram = CountVectorizer(ngram_range=(1, noNGram))
    XTrainNGram = vectorizerNGram.fit_transform(trainSamples)
    
    print vectorizerNGram
    
    
    print len(vectorizerNGram.get_feature_names())
    trainNGramVector = XTrainNGram.toarray()
    print trainNGramVector.shape
    XTestNGram = vectorizerNGram.transform(testSamples)
    testNGramVector = XTestNGram.toarray()
    print testNGramVector.shape
    
    
    XTrainWordFeatures = trainBagVector  # trainNGramVector
    print XTrainWordFeatures.shape
    print trainSentimentSamples.shape
    
    temp = np.column_stack((XTrainWordFeatures, trainSentimentSamples))
    print temp.shape
    XTrainAllFeatures = np.column_stack((temp, trainFreqTweetSamples))
    
    
    XTestWordFeatures = testBagVector  # testNGramVector
    temp = np.column_stack((XTestWordFeatures, testSentimentSamples))
    print temp.shape
    XTestAllFeatures = np.column_stack((temp, testFreqTweetSamples))
    
    
    print XTrainAllFeatures.shape
    
    XEvalAllFeatures = np.column_stack((np.column_stack((evalBagVector, XEvalSentiment)), XEvalFreqTweets))
    
    print XEvalAllFeatures.shape
    
    train = XTrainAllFeatures
    print type(trainBagVector), type(trainStateTransitionVector)
    
    print train.shape
    YTrain = YtrainSamples
    print YTrain.shape
    YTest = YtestSamples
    
    print len(testStateTransitionVector), len(trainStateTransitionVector)
    
    # train = np.hstack([XTrainAllFeatures, XTestAllFeatures])
    train = XTrainAllFeatures
    test = XEvalAllFeatures
    params = {'neighbours':25}
    neighbours = params['neighbours']
    neigh = KNeighborsClassifier(n_neighbors=neighbours)
    YPred = neigh.fit(train, YTrain).predict(test)
    
    print YPred[2:3020]
    
    
    params = {'trees':150, 'criterion':'entropy', 'random_state':None}
    trees = params['trees']
    crit = params['criterion']
    seed = params['random_state']
    clf = RandomForestClassifier(n_estimators=trees, criterion=crit, random_state=seed)
    clf.fit(train, YTrain)
    YPred = clf.predict(test)
    
    params = {'kernel':'rbf'}
    ker = params['kernel']
    YPred = svm.SVC(kernel=ker).fit(train, YTrain).predict(test)
    
    
    lon = np.random.random_integers(-180, 180, 60)
    lat = np.random.random_integers(-90, 90, 60)
    geo_latitude = lat
    geo_longitude = lon
    print type(lat)
    print lat.shape
    GeoPlot(geo_longitude, geo_latitude, YTrain[0:60])
    
    GeoPlot(eval_geo_longitude[0:1000], eval_geo_latitude[0:1000], YPred[0:1000])
    
    GeoSentimentPlot(geo_longitude, geo_latitude, XTrainSentiment[0:60])
    
    print len(eval_geo_longitude)
    eval_geo_longitude = np.array(eval_geo_longitude)
    eval_geo_latitude = np.array(eval_geo_latitude)
    print len(eval_geo_longitude)
    print eval_geo_longitude.shape
    print type(eval_geo_longitude)
    
    GeoSentimentPlot(eval_geo_longitude[0:1000], eval_geo_latitude[0:1000], XEvalSentiment[0:1000])
    
