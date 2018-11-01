from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

class FnnKeras():
    def __init__(self, nnodes_per_hidden_layer=[100], lrate=0.05, momentum=0, batch_size=1):
        self.nnodes_per_hidden_layer = nnodes_per_hidden_layer
        self.lrate = lrate
        self.momentum = momentum
        self.batch_size = batch_size
    
    def fit(self, data, labels, epochs=1):
        """data: ndarray"""
        n_rows = len(data)
        n_attr = len(data[n_rows-1])
        self.model = Sequential()
        # First Hidden Layer
        self.model.add(Dense(units=self.nnodes_per_hidden_layer[0], activation='sigmoid', input_dim=n_attr))
        # 2nd .. Last Hidden Layer
        for i in range(1, len(self.nnodes_per_hidden_layer)):
            self.model.add(Dense(units=self.nnodes_per_hidden_layer[i], activation='sigmoid'))
        # Output Layer
        self.model.add(Dense(units=1, activation='sigmoid'))
        
        sgd = optimizers.SGD(lr=self.lrate, momentum=self.momentum)
        self.model.compile(optimizer=sgd, loss='mean_squared_error')
        self.model.fit(data, labels, batch_size=self.batch_size, epochs=epochs)
        return self

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels, batch_size=self.batch_size)

    def predict(self, sample):
        return self.model.predict(sample, batch_size=self.batch_size)
    