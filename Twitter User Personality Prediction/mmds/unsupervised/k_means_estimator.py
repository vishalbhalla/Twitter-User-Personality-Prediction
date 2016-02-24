import copy
import csv
import logging
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.coo import coo_matrix
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from textblob.blob import TextBlob
from textblob.en.np_extractors import ConllExtractor
from textblob.en.taggers import NLTKTagger
from mmds.utils.time_utils import time_it



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
    Predicted label feature name.
    """
    LABEL_FEATURE_KEY = 'label'

    RELEVENT_FEATURE_LIST = [USER_ID_FEATURE_KEY, LATITUDE_FEATURE_KEY, LONGITUDE_FEATURE_KEY, LABEL_FEATURE_KEY]
    
    def __init__(self, tweet_file_path, no_of_clusters):
        """
        The constructor reads csv file and builds the data matrix.
        """
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
    def perform_clustering(self, features_to_include=None):
        """
        This function performs k-means clustering with "no_of_clusters" clusters of the data present in file at
        "tweet_file_path".
        It returns list of feature vector, where each feature vector contains only "features_to_include" or all features
        if "features_to_include" is None.
        """
        clustering_data_matrix = self.__get_clustering_data_matrix(self.data_matrix)
        transformed_data_matrix = self.vectorizer.fit_transform(clustering_data_matrix)
        
        self.k_means_estimator.fit(transformed_data_matrix, y=None)
        return self.__get_predicted_labels(self.data_matrix, features_to_include)

    @time_it    
    def __get_predicted_labels(self, data_matrix, features_to_include):
        """
        Finds the nearest cluster for all data points and adds a new feature label in all feature vectors of data matrix. The
        data matrix is modified in place.
        It returns a new copy of data_matrix with "features_to_include" features.
        """
        feature_names = self.vectorizer.get_feature_names()
        for feature_vector in data_matrix:
            row = [0] * len(feature_names)
            column = range(len(feature_names))
            data = map(lambda feature_name:feature_vector[feature_name] if feature_name in feature_vector else 0, feature_names)
            feature_csr_matrix = csr_matrix(coo_matrix((data, (row, column))))
            predicted_label = self.k_means_estimator.predict(feature_csr_matrix)
            feature_vector[self.LABEL_FEATURE_KEY] = predicted_label[0]
        
        expanded_data_matrix = self.__get_expanded_data_matrix(data_matrix)
        if features_to_include:
            return self.__get_filtered_data_matrix(expanded_data_matrix, features_to_include)
        else:
            return expanded_data_matrix

    @time_it
    def __get_filtered_data_matrix(self, data_matrix, features_to_include):
        """
        Removes all features except features_to_include
        """
        filtered_data_matrix = []
        for feature_vector in data_matrix:
            filtered_feature_vector = {}
            for feature_name in features_to_include:
                filtered_feature_vector[feature_name] = feature_vector[feature_name]
            filtered_data_matrix.append(filtered_feature_vector)
        return filtered_data_matrix
    
    @time_it
    def __get_expanded_data_matrix(self, data_matrix):
        """
        Adds new keys for missing features to all feature vectors of data_matrix. The data matrix is not modified, but a new 
        modified copy is returned.
        """
        feature_names = self.vectorizer.get_feature_names()
        expanded_data_matrix = copy.deepcopy(data_matrix)
        for feature_vector in expanded_data_matrix:
            for feature_name in feature_names:
                if feature_name not in feature_vector:
                    feature_vector[feature_name] = 0
        return expanded_data_matrix
    
    @time_it
    def predict_labels_for_data(self, file_path, features_to_include=None):
        """
        This function reads the tweets of different users from the file at file_path and assigns the closest 
        cluster center to each user.
        It returns list of tuples of (user_id,predicted_label,latitude, longitude).
        """
        data_matrix = self.__get_data_matrix_from_file(file_path)
        return self.__get_predicted_labels(data_matrix, features_to_include)
        

def write_dict_list_to_csv(dict_list, file_name):
    """
    Saves the list of dictionaries to file at "file_name". Each dictionary should have same set of keys.
    """
    file_writer = csv.DictWriter(open(file_name, "w"), dict_list[0].keys())
    file_writer.writeheader()
    file_writer.writerows(dict_list)
    
if __name__ == "__main__":
    input_file = "../../TwitterData/survey_dump_with_geo_gt_8"
    output_file = "../../TwitterData/k_means_geo_gt_8_out"
    no_of_clusters = 10
    clusterd_data = KMeansEstimator(input_file, no_of_clusters).perform_clustering(KMeansEstimator.RELEVENT_FEATURE_LIST)
    logging.info("Input file:%s, output file:%s, no of clusters:%d", input_file, output_file, no_of_clusters)
    write_dict_list_to_csv(clusterd_data, output_file)
    logging.info("Written predicted labels for %d users in file:%s", len(clusterd_data), output_file)

