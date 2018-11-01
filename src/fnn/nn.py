import numpy as np

MAX_HIDDEN = 10

class NNClassifier:
    def __init__(self, n_nodes=[], momentum=0, batch_size=1, max_iter=100):
        # Checking parameter input
        if (len(n_nodes) > MAX_HIDDEN):
            raise ValueError('Number of hidden layers cannot be greater than {}'.format(MAX_HIDDEN))

        if (not all(x > 0 for x in n_nodes)):
            raise ValueError('Number of nodes in a layer cannot be nonpositive')

        if (batch_size <= 0):
            raise ValueError('Batch size cannot be nonpositive')

        # Setting parameter
        self.n_nodes = n_nodes
        self.n_hiddens = len(n_nodes)
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.weights = []
        self.prev_weights = []

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __feed_forward(self, x):
        outputs = [x]
        for weight in self.weights:
            outputs.append(self.__sigmoid(outputs[-1] @ weight))

        del outputs[0]
        return outputs

    def __update_weights(self, data, target):
        # Do forward propagation
        for x, y in zip(data, target):
            res = self.__feed_forward(x)

    def fit(self, data, target):
        # Initialize weights with random numbers
        n_features = data.shape[1]
        self.prev_weights = self.weights.copy()
        if (self.n_hiddens > 0):
            self.weights = [np.random.randn(n_features, self.n_nodes[0])]
            for i in range(1, self.n_hiddens):
                self.weights.append(np.random.randn(self.n_nodes[i-1], self.n_nodes[i]))
            self.weights.append(np.random.randn(self.n_nodes[self.n_hiddens - 1], 1))
        else:
            self.weights = [np.random.randn(n_features, 1)]

        for _ in range(self.max_iter):
            # Random shuffle data and target simultaneously
            p = np.random.permutation(data.shape[0])
            data, target = data[p], target[p]

            # Do gradient descent per batch
            for i in range(0, data.shape[0], self.batch_size):
                index = list(range(i, i+self.batch_size))
                self.__update_weights(data[index], target[index])