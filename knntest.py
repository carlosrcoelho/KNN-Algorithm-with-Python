import numpy as np
from sklearn import datasets  # import datasets from sklearn
from sklearn.model_selection import train_test_split  # this is used to split the data
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter  # this is used to count the number of labels
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # this is used to color the data points


iris = datasets.load_iris()  # load the iris dataset from sklearn
X, y = iris.data, iris.target  # assign the data and target to X and y respectively

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)  # split the data into train and test sets


# # print the shapes of the training and test sets
# print(X_train.shape)  # number of samples (120) and features (4) in the training set
# print(X_train[0])  # print the first sample in the training set

# print(y_train.shape)  # number of samples (120) in the training set
# print(y_train)  # print the labels of the training set

# # plot the data
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()


# # how to use most_common function from Counter
# a = [1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8]
# most_common = Counter(a).most_common(1)
# print(most_common)  # [(1, 3). The most common element is 1, which appears 3. 

# Using the KNN class
from knn import KNN
clf = KNN(k=3)  # create a KNN classifier
clf.fit(X_train, y_train)  # train the classifier
predictions = clf.predict(X_test)  # make predictions on the test set

acc = np.sum(predictions == y_test) / len(y_test)  # compute the accuracy

print(f'Accuracy model: {acc * 100:.2f}%')
