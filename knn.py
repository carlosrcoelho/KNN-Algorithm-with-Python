import numpy as np
from collections import Counter # this is used to count the number of labels

def euclidean_distance(x1, x2):  # this function computes the euclidean distance between two vectors
    return np.sqrt(np.sum((x1 - x2)**2))   # this is the euclidean distance formula

# create a KNN class
class KNN:
    def __init__(self, k):  # this is the constructor
        self.k = k
    
    def fit(self, X, y):  # this is the training function
        self.X_train = X  # store the training data
        self.y_train = y  # store the training labels

    def predict(self, X):  # this is the main function
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):  # this is a helper function
        # compute the distances between x and all samples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get the k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]  # get the indices of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
