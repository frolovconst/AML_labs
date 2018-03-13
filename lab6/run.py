import numpy as np
from sklearn.metrics import f1_score
from utils import load_data, train_test_split

X, Y = load_data("iris.csv")

X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)


class KNN:
    def __init__(self):
        """
        Your initialization procedure if required
        """
        pass

    def fit(self,X,Y):
        """
        KNN algorithm in the simples implementation can work only with
        continuous features

        X: training data, numpy array of shape (n,m)
        Y: training labels, numpy array of shape (n,1)
        """

        # Hint: make sure the data passed as input are of the type float
        # Hint: make sure to create copies of training data, not copies of
        #       references
        pass

    def predict(self, X, nn=5):
        """
        X: data for classification, numpy array of shape (k,m)
        nn: number of nearest neighbours that determine the final decision

        returns
        labels: numpy array of shape (k,1)
        """
        # Hint: make sure the data passed as input are of the type float
        pass


# Task:
# 1. Implement function fit in the class KNN
# 2. Implement function predict in the class KNN, where neighbours are weighted
#     according to uniform weights
# 3. Test your algorithm on iris dataset according to
#     f1_score (expected: 0.93)
# 4. Test your algorithm on mnist_small dataset according to
#     f1_score (expected: 0.7)
# 5. Test your algorithm on mnist_large dataset according to
#     f1_score (expected: 0.86)
# 6. Implement function predict in the class KNN, where neighbours are weighted
#     according to their distance to the query instance

np.random.seed(1)

c = KNN()
c.fit(X_tr,Y_tr)
label_p = c.predict(X_t)

print("Test score %.2f"%(None))
