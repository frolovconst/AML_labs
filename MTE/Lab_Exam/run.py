from pandas import read_csv, get_dummies, concat
from numpy import mean, dtype
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# In this assignment you are going to work with Pandas
# what is assessed is your ability to work with documentation
# overall you need to write 5 lines of code
# Go through the following code line by line and complete
# the functions fill_na and embed_categories

#        Taks Description
# In current task you are going to do data pre-processing for KNN algorithm.
# KNN with euclidean distance metric is designed to work only with numerical features
# You are going to test KNN on modified Titanic dataset. You can get familiar with
# dataset header in the file titanic_modified.csv

# You are going to complete two types of pre-processing:
# 1. Eliminating Null (N/A) values
# 2. Mapping categorical features to binary indicator features

def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - pandas DataFrame of shape (n,m) of input features
             Y - numpy array of output features of shape (n,)
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header=None)
    X = data.loc[:, data.columns[:-1] ]
    Y = data.loc[:, data.columns[-1]  ]
    return X, Y.as_matrix()


X, Y = load_data("titanic_modified.csv")

def fill_na(X):
    """
    Iterates over columns of DataFrame X, any N/A values replaced by the most
    frequent element in the column
    :param X: DataFrame with input features
    :return: copy of the DataFrame with N/A replaced by the most frequent elements
    """
    # Just to be safe, create a copy of current DataFrame
    X = X.copy()
    for i in range(X.shape[1]):
        col_name = X.columns[i]         # current column name
        is_null = X[col_name].isnull()  # check for N/A values
        if is_null.any():
            # Find the most frequent element value in the current column
            # Hint: use the function mode from scipy.stats
            # Hint: input slice for mode function can be obtained as
            #       X.loc[~is_null, col_name]
            # Hint: be careful with N/A values
            #None  # <- Replace this line
            mfv, f = mode(X.loc[~is_null,col_name])
#             print(mfv)
#             print(where(is_null))
            # Replace N/A entries with most_common value
            # Hint: slice DataFrame with X.loc[???]
            # Hint: make use of is_null when slicing
#             None  # <- Replace this line
            X.loc[is_null, col_name] = mfv
    return X

def embed_categories(X):
    """
    Replaces columns with categorical features by binary feature indicator columns
    :param X: DataFrame with input features
    :return: DataFrame with binary feature indicators
    """
    X = X.copy()
    n_features = X.shape[1]
    c_feature = 0

    while c_feature < n_features:

        col_name = X.columns[c_feature]     # Get current column name
        current_type = X.dtypes[c_feature]  # Get current column type

        if current_type == dtype(object):
            # Your goal is to transform the values of categorical features
            # into binary feature indicators.
            # This procedure requires to transform a data sample of the format
            # Specie | Height | Weight
            #    Cat |     30 |      3
            #    Dog |     50 |     10
            #
            # Into the format
            # Height | Weight | Cat | Dog
            #     30 |      3 |   1 |   0
            #     50 |     10 |   0 |   1
            #
            # Luckily Pandas can do the hardest part for you with the function get_dummies
            # Hint: use get_dummies from Pandas. The input should be the current column
            crnt_features = get_dummies(X[col_name])
            # Drop the column you have just transformed
            # Since you replace features with binary indicators,
            # old features is no use for you any more
            # Hint: use drop from Pandas.DataFrame to remove current column
            X.drop(columns=[col_name], inplace=True)
            # Concatenate the rest of the DataFrame with new columns
            # Hint: use function concat from Pandas
            # Hint: Pay attention to the argument axis
            X = concat((X, crnt_features), axis=1)

            n_features -= 1
        else:
            c_feature += 1
    # Cast DataFrame to matrix
    return X.as_matrix()

def k_fold_validation(X, Y):
    kf = KFold(n_splits=50)
    fold_score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        fold_score.append(f1_score(y_test, y_pred))
    return mean(fold_score)

X = fill_na(X)
X = embed_categories(X)

# Test the performance with one hot categorical feature embeddings
s1 = k_fold_validation(X, Y)
print("Score for original features: ", s1)

# After checking the performance of this classification procedure
# apply a dimensionality reduction technique (PCA). It several
# purposes:
# 1. Implicit data normalization (PCA pre-processing)
# 2. Feature orthogonalization
# 3. Dimensionality reduction

# Apply Principal Component Analysis
num_components = X.shape[1]-2      # reduce features dimension by 2
pca = PCA(n_components=num_components)
X_tran = pca.fit_transform(X)

# Test the performance with transformed features
s2 = k_fold_validation(X_tran, Y)
print("Score for transformed features: ", s2)
print("Gain: ", s2 - s1)           # You should observe gain around 1-3%