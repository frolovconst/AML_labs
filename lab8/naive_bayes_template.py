import numpy as np
import math
from pandas import get_dummies, DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold
from lab8.utils import load_data


class NaiveBayes:

    def __init__(self):
        """
        Your initialization procedure if required
        """
        pass

    def fit(self, X, Y):
        """
        This method calculates class probabilities and conditional probabilities to be used for prediction

        Both numerical and categorical features are accepted.
        Conditional probability of numerical features is calculated based on Probability Density Function
        (assuming normal distribution)

        :param X: training data, numpy array of shape (n,m)
        :param Y: training labels, numpy array of shape (n,1)
        """
        # TODO START YOUR CODE HERE

        # END YOUR CODE HERE

    @staticmethod
    def estimate_mean_and_stdev(values):
        """
        Estimates parameters of normal distribution - empirical mean and standard deviation
        :param values: attribute sample values
        :return: mean, stdev
        """
        # TODO START YOUR CODE HERE

        # END YOUR CODE HERE

    @staticmethod
    def calc_probability(val, mean, stdev):
        """
        Estimates probability of encountering a point (val) given parameters of normal distribution
        based on probability density function
        :param val: point
        :param mean: mean value
        :param stdev: standard deviation
        :return: relative likelihood of a point
        """
        # TODO START YOUR CODE HERE

        # END YOUR CODE HERE

    def predict(self, X):
        """
        Predict class labels for given input. Refer to lecture slides for corresponding formula
        :param X: test data, numpy array of shape (n,m)
        :return: numpy array of predictions
        """
        # TODO START YOUR CODE HERE

        # END YOUR CODE HERE

    def get_params(self, deep = False):
        return {}


X, Y = load_data("crx.data.csv")
# indexes of numerical attributes
numerical_attrs = [1, 2, 7, 10, 13, 14]
X[:, numerical_attrs] = X[:, numerical_attrs].astype(float)

# categorical features only. Use this to test your initial implementation
X_cat = np.delete(X, numerical_attrs, 1)
scores = cross_val_score(NaiveBayes(), X_cat, Y, cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Categorical Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# use this as a benchmark. Your algorithm (on categorical features) should reach the same accuracy
X_dummy = DataFrame.as_matrix(get_dummies(DataFrame(X_cat)))
scores = cross_val_score(MultinomialNB(), X_dummy, Y.ravel(), cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Categorical Accuracy of Standard NB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# all (mixed) features. Use this to test your final implementation
scores = cross_val_score(NaiveBayes(), X, Y.ravel(), cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Overall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# write your thoughts here (if any)