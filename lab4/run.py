from pandas import read_csv
import numpy as np


def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer', dtype=str)
    else:
        data = read_csv(path_to_csv, header=None, dtype=str)
    data.fillna('', inplace=True)
    data = data.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class DTree:
    """
    Simple decision tree classifier for a training data with categorical
    features
    """
    _model = None

    def fit(self, X, Y):
        self._model = create_branches({'attr_id': -1,
                                       'branches': dict(),
                                       'decision': None}, X, Y)

    def predict(self, X):
        raise NotImplementedError

        if X.ndim == 1:
            return None
        elif X.ndim == 2:
            return None
        else:
            print("Dimensions error")

    def prune(self):
        """
        Implement pruning to improve generalization
        """
        raise NotImplementedError


def elem_to_freq(values):
    """
    input: numpy array
    returns: The counts of unique elements, unique elements are not returned
    """
    # hint: check numpy documentation for how to count unique values
    raise NotImplementedError
    return None


def entropy(elements):
    """
    Calculates entropy of a numpy array of instances
    input: numpy array
    returns: entropy of the input array based on the frequencies of observed
             elements
    """
    # hint: use elem_to_freq(arr)
    raise NotImplementedError
    return None


def information_gain(A, S):
    """
    input:
        A: the values of an attribute A for the set of training examples
        S: the target output class

    returns: information gain for classifying using the attribute A
    """
    # hint: use entropy(arr)
    raise NotImplementedError
    return gain


def choose_best_attribute(X, Y):
    """
    input:
        X: numpy array of size (n,m) containing training examples
        Y: numpy array of size (n,) containing target class

    returns: the index of the attribute that results in maximum information
             gain. If maximum information gain is less that eps, returns -1
    """

    eps = 1e-10

    raise NotImplementedError
    return best_attr


def most_common_class(Y):
    """
    input: target class values
    returns: the value of the most common class
    """
    raise NotImplementedError
    return None


def create_branches(node, X, Y):
    """
    create branches in a decision tree recursively
    input:
        node: current node represented by a dictionary of format
                {'attr_id': -1,
                 'branches': dict(),
                 'decision': None},
              where attr_id: specifies the current attribute index for branching
                            -1 mean the node is leaf node
                    braches: is a dictionary of format {attr_val:node}
                    decision: contains either the best guess based on
                            most common class or an actual class label if the
                            current node is the leaf
        X: training examples
        Y: target class

    returns: input node with fields updated
    """
    # choose best attribute to branch
    attr_id = None
    node['attr_id'] = attr_id
    # record the most common class
    node['decision'] = None

    if attr_id != -1:
        # find the set of unique values for the current attribute
        attr_vals = None

        for a_val in attr_vals:
            # compute the boolean array for slicing the data for the next
            # branching iteration
            # hint: use logical operation on numpy array
            # for more information about slicing refer to numpy documentation
            sel = None
            # perform slicing
            X_branch = X[sel, :]
            Y_branch = Y[sel]
            node_template = {'attr_id': -1,
                             'branches': dict(),
                             'decision': None}
            # perform recursive call
            node['branches'][a_val] = None
    return node


def traverse(model,sample):
    """
    recursively traverse decision tree
    input:
        model: trained decision tree
        sample: input sample to classify

    returns: class label
    """
    if model['attr_id'] == -1:
        decision = model['decision']
    else:
        attr_val = sample[ model['attr_id'] ]
        if attr_val not in model['branches']:
            decision = model['decision']
        else:
            decision = traverse(model['branches'][attr_val], sample)
    return decision


def train_test_split(X, Y, fraction):
    """
    perform the split of the data into training and testing sets
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        fraction: number between 0 and 1, specifies the size of the training
                data

    returns:
        X_train
        Y_train
        X_test
        Y_test
    """
    if fraction < 0 or fraction > 1:
        raise Exception("Fraction for split is not valid")

    # do random sampling for splitting the data
    raise NotImplementedError
    return None


def measure_error(Y_true, Y_pred):
    """
    returns an error measure of your choice
    """
    return None


def recall(Y_true, Y_pred):
    """
    returns recall value
    """
    return None


# 1.  test your implementation on data_1.csv
#     refer to lecture slides to verify the correctness
# 2.  test your implementation on mushrooms_modified.csv
# 3.  test your implementation on titanic_modified.csv


X,Y = load_data("data_1.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree()
d_tree.fit(X_train,Y_train)
Y_pred = d_tree.predict(X_test)
print("Correctly classified: %.2f%%" % (measure_error(Y_test,Y_pred) * 100))
print("Recall %.4f" % recall(Y_test, Y_pred))
