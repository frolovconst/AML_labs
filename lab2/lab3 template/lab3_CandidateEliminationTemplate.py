from pandas import read_csv
import numpy as np


def load_data(path_to_csv, has_header=True):
    if has_header:
        data = read_csv(path_to_csv, header='infer')
    else:
        data = read_csv(path_to_csv, header=None)
    data = data.as_matrix()
    X = data[:, 0:-1]
    Y = data[:, -1]
    return X, Y


class CandidateElimination:

    # candidate elimination algorithm
    def fit(self, training_data, labels):
        S = self.initialize_to_first_positive(training_data, labels)
        G = self.initialize_to_most_general(training_data)
        training_examples = len(training_data)
        for i in range(training_examples):
            if labels[i] == "yes":
                G = [g for g in G if self.is_consistent(training_data[i], g, True)]
                not_consistent = [s for s in S if not self.is_consistent(training_data[i], s, True)]
                S = [s for s in S if self.is_consistent(training_data[i], s, True)]
                for n in not_consistent:
                    self.add_min_generalization(n, training_data[i], G, S)
                S = [s for s in S if not self.is_more_general_than_any(s, S)]
            else:
                S = [s for s in S if self.is_consistent(training_data[i], s, False)]
                not_consistent = [g for g in G if not self.is_consistent(training_data[i], g, False)]
                G = [g for g in G if self.is_consistent(training_data[i], g, False)]
                for n in not_consistent:
                    self.add_min_specialization(n, training_data[i], S, G, training_data)
                G = [g for g in G if not self.is_less_general_than_any(g, G)]
        print("Final Version Space:")
        print("S: ", S)
        print("G: ", G)

    def initialize_to_first_positive(self, training_data, labels):
        """"
        Returns list with one hypothesis which is equal to the first positive example
        """
        for i in range(len(labels)):
            if labels[i] == 'yes':
                init_set = [training_data[i, :]]
                return init_set

    def initialize_to_most_general(self, training_data):
        """"
        Returns list with one most general hypothesis - ['?', '?', '?', '?'...]
        """
        hypothesis = []
        for i in range(training_data.shape[1]):
            hypothesis.append("?")
        return [np.array(hypothesis, dtype=object)]

    def is_consistent(self, training_example, hypothesis, is_positive):
        """"
        Returns True if the hypothesis classifies the training_example as:
            - positive if it's positive
            - negative if it's negative
        """
        # %%% TODO START YOUR CODE HERE %%%


        # %%% END YOUR CODE HERE %%%


    def add_min_generalization(self, hypothesis, training_example, G, S):
        """
        Makes the hypothesis consistent with training_example
        Adds it to S if some member of G is more general
        """
        # %%% TODO START YOUR CODE HERE %%%


        # %%% END YOUR CODE HERE %%%

    def is_more_general_than_any(self, hypothesis, set):
        """
        Checks if the hypothesis is more general than any hypothesis in the set
        """
        for h in set:
            if self.is_more_general(hypothesis, h):
                return True
        return False

    def add_min_specialization(self, hypothesis, training_example, S, G, training_data):
        """
        Generates all possible minimal specializations by replacing '?' by all possible values except for value in negative example
        Adds each such specialization to G if some member of S is more specific than the specialization
        """
        # %%% TODO START YOUR CODE HERE %%%


        # %%% END YOUR CODE HERE %%%


    def is_less_general_than_any(self, hypothesis, set):
        """
        Checks if the hypothesis is less general than any hypothesis in the set
        """
        for h in set:
            if self.is_more_general(h, hypothesis):
                return True
        return False

    def is_more_general(self, hypothesis1, hypothesis2):
        """
        Returns True if hypothesis1 is more general than hypothesis2
        """
        # %%% TODO START YOUR CODE HERE %%%


        # %%% END YOUR CODE HERE %%%


    def is_equal(self, hypothesis1, hypothesis2):
        """
        Returns True if hypotheses are equal
        """
        for i in range(len(hypothesis1)):
            if hypothesis1[i] != hypothesis2[i]:
                return False
        return True


CE = CandidateElimination()

X, Y = load_data("data_1.csv")
CE.fit(X, Y)
