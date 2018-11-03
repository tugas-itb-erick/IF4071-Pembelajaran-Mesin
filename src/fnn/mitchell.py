from copy import deepcopy

import numpy as np

class FnnClassifier:
    def __init__(self, n_nodes=[], lrate=0.05, momentum=0, batch_size=1, max_iter=100):
        print(self.__MAX_HIDDEN)
        # Checking parameter input
        if (len(n_nodes) > self.__MAX_HIDDEN):
            raise ValueError('Number of hidden layers cannot be greater than {}'.format(self.__MAX_HIDDEN))

        if (not all(x > 0 for x in n_nodes)):
            raise ValueError('Number of nodes in a layer cannot be nonpositive')

        if (batch_size <= 0):
            raise ValueError('Batch size cannot be nonpositive')

        # Setting parameter
        self.n_nodes = n_nodes
        self.n_hiddens = len(n_nodes)
        self.lrate = lrate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.weights = []
        self.prev_weights = []

    @property
    def __MAX_HIDDEN(self):
        return 10

    def __stochastic_gradient_descend(self, data, target):
        for x, y in zip(data, target):
            values_layers = self.__feed_forward(x)
            errors_layers = self.__backward_prop(y, values_layers)
            values_layers.insert(0, x)

            # Update weight
            new_weights = []
            for ilayer, (weights_per_layer, prev_weights_per_layer) in enumerate(zip(self.weights, self.prev_weights)):
                new_weights_per_layer = []
                for inode, (weight_all, prev_weight_all) in enumerate(zip(weights_per_layer, prev_weights_per_layer)):
                    new_weight_all = []
                    for iweight, (weight, prev_weight) in enumerate(zip(weight_all, prev_weight_all)):
                        new_weight_all.append(self.__calculate_weight(weight, prev_weight, 
                        values_layers[ilayer][inode], errors_layers[ilayer][iweight]))
                    new_weights_per_layer.append(new_weight_all)
                new_weights.append(np.array(new_weights_per_layer))
            self.prev_weights = deepcopy(self.weights)
            self.weights = new_weights
    
    def __feed_forward(self, x):
        outputs = [x]
        for weight in self.weights:
            outputs.append(self.__sigmoid(outputs[-1] @ weight))
        del outputs[0]
        return outputs
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __backward_prop(self, target, values_layers):
        n_hiddens_out_layers = len(values_layers)
        errors_layers = [None] * n_hiddens_out_layers
        for i in range(n_hiddens_out_layers-1, 0-1, -1):
            errors = []
            if i < n_hiddens_out_layers-1: # (hidden layer)
                for inode, output in enumerate(values_layers[i]):
                    errors.append(self.__hidden_error(output, inode, i, errors_layers))
            else: # i == n_hiddens_out_layers-1 (output layer)
                for output in values_layers[i]:
                    errors.append(self.__output_error(output, target))
            errors_layers[i] = np.array(errors)
        return errors_layers

    def __output_error(self, output, target):
        return output * (1 - output) * (target - output)
    
    def __hidden_error(self, output, inode, index_layer, errors_layers):
        index_delta = index_layer + 1
        index_weight = index_layer + 1
        sigma = 0
        for i in range(0, len(self.weights[index_weight][inode])):
            # takut salah indexnya
            sigma += self.weights[index_weight][inode][i] * errors_layers[index_delta][i]
        return output * (1 - output) * sigma

    def __calculate_weight(self, weight, prev_weight, err, val):
        return weight + self.momentum * prev_weight + self.lrate * err * val

    def fit(self, data, target):
        self.__initialize_weights(data)
        print(self.weights)

        for _ in range(self.max_iter):
            # Random shuffle data and target simultaneously
            p = np.random.permutation(data.shape[0])
            data, target = data[p], target[p]

            # Do gradient descent per batch
            for i in range(0, data.shape[0], self.batch_size):
                index = list(range(i, i+self.batch_size))
                self.__stochastic_gradient_descend(data[index], target[index])
            
        print(self.weights)
        return self

    def __initialize_weights(self, data):
        # Initialize weights with random numbers
        n_features = data.shape[1]
        if (self.n_hiddens > 0):
            self.weights = [np.random.randn(n_features, self.n_nodes[0])]
            for i in range(1, self.n_hiddens):
                self.weights.append(np.random.randn(self.n_nodes[i-1], self.n_nodes[i]))
            self.weights.append(np.random.randn(self.n_nodes[self.n_hiddens - 1], 1))
        else:
            self.weights = [np.random.randn(n_features, 1)]
        
        # Assume first prev_weights be zeroes
        self.prev_weights = deepcopy(self.weights)
        for i, prev_weight_per_layer in enumerate(self.prev_weights):
            for j, prev_weight_all in enumerate(prev_weight_per_layer):
                for k, _ in enumerate(prev_weight_all):
                    self.prev_weights[i][j][k] = 0
