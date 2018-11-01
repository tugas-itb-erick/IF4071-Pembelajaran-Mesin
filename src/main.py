from fnn.keras import FnnKeras
import pandas as pd

data_pd = pd.read_csv("iris.data.txt", header=None)
labels = data_pd[4].values
del data_pd[4]
data = data_pd.values

model = FnnKeras([10, 20]).fit(data, labels, epochs=25)
print(model.evaluate(data, labels))
print(model.predict(data))
