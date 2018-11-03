from fnn.keras import FnnKeras
from fnn.mitchell import FnnClassifier
import pandas as pd

data_pd = pd.read_csv("iris.data.txt", header=None)
labels = data_pd[4].values
del data_pd[4]
data = data_pd.values

model_keras = FnnKeras([10, 20]).fit(data, labels, epochs=1)
# print(model_keras.evaluate(data, labels))
# print(model_keras.predict(data))

model = FnnClassifier([10, 20], max_iter=25).fit(data, labels)
# print(model.evaluate(data, labels))
# print(model.predict(data))
