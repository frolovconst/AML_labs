
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from utils import load_data, train_test_split


# In[166]:

X, Y = load_data("iris.csv")
X = X.astype(np.float)
# Y = Y.astype(np.float)
X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)


# In[160]:

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
        self.X = X.copy()
        self.Y = Y.copy()

#     def euclidean()
    
    def predict(self, X, nn=5):
        """
        X: data for classification, numpy array of shape (k,m)
        nn: number of nearest neighbours that determine the final decision

        returns
        labels: numpy array of shape (k,1)
        """
        # Hint: make sure the data passed as input are of the type float
        result = np.array([[]], dtype=np.object)
        for x in X:
            distances = (self.X-x)
            distances = np.sqrt(np.einsum('ij,ij->i', distances, distances))
            nearest = np.argsort(distances)[:nn]
#             print(nearest)
#             print(self.Y[nearest])
            unique,pos = np.unique(self.Y[nearest],return_inverse=True)
            if unique.size==1:
                result = np.append(result, np.array([[unique[0]]]))
#             else:
#                 counts = np.bincount(pos)                     
#                 maxpos = counts.argmax()  
#                 result = np.append(result, np.array([[unique[maxpos]]]))
            else:
                dct = dict()
                for each in nearest:
                    if self.Y[each,0] in dct:
                        dct[self.Y[each,0]] += 1/distances[each]
                    else:
                        dct[self.Y[each,0]] = 1/distances[each]
                result = np.append(result, np.array([[max(dct, key=dct.get)]]))
                
        return result


# In[ ]:

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


# In[167]:

np.random.seed(1)

c = KNN()
c.fit(X_tr,Y_tr)


# In[168]:

label_p = c.predict(X_t)[:,np.newaxis]


# In[169]:

print("Test score %.2f"%(f1_score(Y_t.flatten(), label_p.flatten(), average='weighted')))


# In[170]:

X, Y = load_data("mnist_small.csv")
X = X.astype(np.int)
# Y = Y.astype(np.float)
X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)

np.random.seed(1)

c = KNN()
c.fit(X_tr,Y_tr)

label_p = c.predict(X_t)[:,np.newaxis]
label_p = label_p.astype(np.int)
Y_t = label_p.astype(np.int)

print("Test score %.2f"%(f1_score(Y_t.flatten(), label_p.flatten(), average='weighted')))


# In[171]:

X, Y = load_data("mnist_large.csv")
X = X.astype(np.int)
# Y = Y.astype(np.float)
X_tr, Y_tr, X_t, Y_t = train_test_split(X,Y,.7)

np.random.seed(1)

c = KNN()
c.fit(X_tr,Y_tr)

label_p = c.predict(X_t)[:,np.newaxis]
label_p = label_p.astype(np.int)
Y_t = label_p.astype(np.int)

print("Test score %.2f"%(f1_score(Y_t.flatten(), label_p.flatten(), average='weighted')))

