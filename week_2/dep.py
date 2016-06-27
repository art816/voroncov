import pandas
from sklearn import cross_validation
from sklearn import neighbors
import sklearn
import numpy
import operator


data = pandas.read_csv('wine.data', header=None)
data_classes = pandas.read_csv('wine.data', header=None, usecols=[0])
data_features = pandas.read_csv('wine.data', header=None, usecols=range(1,14))
cr_valid = cross_validation.KFold(n=data_features.count()[1], n_folds=5,
                                  random_state=42, shuffle=True)

cross_score=dict()
for n_neighbors in range(1,51):
    KNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    cross_score[n_neighbors] = numpy.mean(sklearn.cross_validation.cross_val_score(
        KNeighborsClassifier, data_features, data_classes[0].values,
        cv=cr_valid, scoring='accuracy'))

sorted_cross_score = sorted(cross_score.items(), key=operator.itemgetter(1), reverse=True)
print("best n and cross_score", sorted_cross_score[0])

preprocessing_data_features = sklearn.preprocessing.scale(data_features)
for n_neighbors in range(1,51):
    KNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors)
    cross_score[n_neighbors] = numpy.mean(sklearn.cross_validation.cross_val_score(
        KNeighborsClassifier, preprocessing_data_features, data_classes[0].values,
        cv=cr_valid, scoring='accuracy'))

sorted_cross_score = sorted(cross_score.items(), key=operator.itemgetter(1), reverse=True)
print("best n and cross_score", sorted_cross_score[0])

