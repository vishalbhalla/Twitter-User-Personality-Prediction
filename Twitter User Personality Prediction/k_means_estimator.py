from collections import Counter
import copy
import csv
from scipy.sparse import csr_matrix
from scipy.sparse.coo import coo_matrix
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import sys
from textblob.blob import TextBlob
from textblob.en.np_extractors import ConllExtractor
from textblob.en.taggers import NLTKTagger

from time_utils import time_it
import logging


class KMeansEstimator:
    """
    This class reads the tweets of users from a file and builds cluster centers on that data. It also provides
    method for finding the closest cluster center of unseen data.
    """
    
    ADJECTIVE = 'JJ'
    
    """
    Feature keys used in clustering...
    """
    POLARITY_FEATURE_KEY = 'polarity'
    SUBJECTIVITY_FEATURE_KEY = 'subjectivity'
    TWEET_COUNT_FEATURE_KEY = 'tweetCount'
    """
    Features not considered for clustering...
    """
    USER_ID_FEATURE_KEY = 'userId'
    LONGITUDE_FEATURE_KEY = 'longitude'
    LATITUDE_FEATURE_KEY = 'latitude'
    
    """
    The constructor reads csv file and builds the data matrix.
    """
    def __init__(self, tweet_file_path, no_of_clusters):
        self.np_extractor = ConllExtractor()
        self.pos_tagger = NLTKTagger()
        self.tweet_file_path = tweet_file_path
        self.data_matrix = self.__get_data_matrix_from_file(tweet_file_path)
        self.vectorizer = DictVectorizer(sparse=True)
        self.k_means_estimator = KMeans(init="random", n_clusters=no_of_clusters)
        
    @time_it
    def __get_data_matrix_from_file(self, tweet_file_path):
        """
        Reads tweets from csv file at path "tweet_file_path", extracts features from the tweets and returns list
        of all feature vectors.
        """
        file_reader = csv.reader(open(tweet_file_path, "rb"), delimiter=',')
        next(file_reader)
        data_matrix = []
        for row in file_reader:
            logging.info("Extracting features for user_id:%s", row[0])
            feature_vector = {}
            feature_vector[self.USER_ID_FEATURE_KEY] = int(row[0])
            feature_vector[self.LATITUDE_FEATURE_KEY] = float(row[2])
            feature_vector[self.LONGITUDE_FEATURE_KEY] = float(row[3])
            feature_vector[self.TWEET_COUNT_FEATURE_KEY] = int(row[4])
            feature_vector.update(self.__get_features_from_tweet_text(row[1].decode('utf-8')))
            data_matrix.append(feature_vector)
            logging.info("Successfully extracted features for user_id:%s", row[0])
        return data_matrix
    
    @time_it
    def __get_features_from_tweet_text(self, tweet_text):
        """This function returns the following features from the tweet text:
        - Adjectives and their corresponding frequencies found in the tweet. Each adjective is a separate feature.
        - Subjectivity and polarity as determined by TextBlob.
        :returns:  (key,value) map of all features found. 
        """
        text_blob = TextBlob(tweet_text, np_extractor=self.np_extractor, pos_tagger=self.pos_tagger);
        adjective_map = dict(Counter((ele[0] for ele in set(text_blob.pos_tags) if ele[1] == self.ADJECTIVE)))
        polarity = text_blob.sentiment[0]
        subjectivity = text_blob.sentiment[1]
        return dict(adjective_map.items() + {self.POLARITY_FEATURE_KEY:polarity, self.SUBJECTIVITY_FEATURE_KEY:subjectivity}.items())
    
    @time_it
    def __get_clustering_data_matrix(self, data_matrix):
        """
        This method removes unnecessary features(features like user_id which are not relevant for building cluster centers) from
        the data matrix and returns a copy of the data matrix.
        """
        data_matrix_copy = copy.deepcopy(data_matrix)
        for feature_vector in data_matrix_copy:
            feature_vector.pop(self.USER_ID_FEATURE_KEY)
            feature_vector.pop(self.LATITUDE_FEATURE_KEY)
            feature_vector.pop(self.LONGITUDE_FEATURE_KEY)
        return data_matrix_copy


    @time_it
    def perform_clustering(self):
        """
        This function performs k-means clustering with "no_of_clusters" clusters of the data present in file at
        "tweet_file_path".
        """
        clustering_data_matrix = self.__get_clustering_data_matrix(self.data_matrix)
        transformed_data_matrix = self.vectorizer.fit_transform(clustering_data_matrix)
        
        self.k_means_estimator.fit(transformed_data_matrix, y=None)
        return self.__get_predicted_labels(self.data_matrix)

    @time_it    
    def __get_predicted_labels(self, data_matrix):
        predicted_labels = []
        feature_names = self.vectorizer.get_feature_names()
        for feature_vector in data_matrix:
            user_id = feature_vector.pop(self.USER_ID_FEATURE_KEY)
            latitude = feature_vector.pop(self.LATITUDE_FEATURE_KEY)
            longitude = feature_vector.pop(self.LONGITUDE_FEATURE_KEY)
            row = [0] * len(feature_names)
            column = range(len(feature_names))
            data = map(lambda feature_name:feature_vector[feature_name] if feature_name in feature_vector else 0, feature_names)
            feature_csr_matrix = csr_matrix(coo_matrix((data, (row, column))))
            predicted_label = self.k_means_estimator.predict(feature_csr_matrix)
            predicted_labels.append((user_id, predicted_label[0], latitude, longitude))
        return predicted_labels;
    
    @time_it
    def get_labels_for_data(self, file_path):
        """
        This function reads the tweets of different users from the file at file_path and assigns the closest 
        cluster center to each user.
        It returns list of tuples of (user_id,predicted_label,latitude, longitude).
        """
        data_matrix = self.__get_data_matrix_from_file(file_path)
        return self.__get_predicted_labels(data_matrix)
    
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    no_of_clusters = int(sys.argv[3])
    clusterd_data = KMeansEstimator(input_file, no_of_clusters).perform_clustering()
    file_writer = csv.writer(open(output_file, "w"), delimiter=",")
    logging.info("Input file:%s, output file:%s, no of clusters:%d", input_file, output_file, no_of_clusters)
    file_writer.writerow(['user_id', 'label', 'latitude', 'logitude'])
    [file_writer.writerow(row) for row in clusterd_data]
    logging.info("Written predicted labels for %d users in file:%s", len(clusterd_data), output_file)

