import pandas
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
import numpy
import operator
from collections import OrderedDict




train_data = pandas.read_csv('perceptron-train.csv', header=None)
test_data = pandas.read_csv('perceptron-test.csv', header=None)

Perceptron = linear_model.Perceptron(random_state=241)
Perceptron.fit(train_data[[1, 2]], train_data[0])
accuracy = metrics.accuracy_score(
    test_data[0], Perceptron.predict(test_data[[1, 2]]))
print('accuracy', accuracy)

Scaler = preprocessing.StandardScaler()
scaled_train_data = Scaler.fit_transform(train_data[[1,2]])
scaled_test_data = Scaler.transform(test_data[[1,2]])

Perceptron = linear_model.Perceptron(random_state=241)
Perceptron.fit(scaled_train_data, train_data[0])
scaled_accuracy = metrics.accuracy_score(
    test_data[0], Perceptron.predict(scaled_test_data))
print('scaled_accuracy', scaled_accuracy)

print('deff_accuracy', scaled_accuracy - accuracy)
