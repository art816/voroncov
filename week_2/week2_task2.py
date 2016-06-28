import pandas
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
import numpy
import operator
from collections import OrderedDict

data = datasets.load_boston()
preprocessing_data = preprocessing.scale(data.data)
numpy.linspace(1,10,200)
cr_valid = cross_validation.KFold(n=numpy.size(data.data, 0), n_folds=5,
                                  random_state=42, shuffle=True)
cross_score=OrderedDict()
for p in numpy.linspace(1,10,200):
    KNeighborsRegressor = neighbors.KNeighborsRegressor(
        metric='minkowski', p=p, n_neighbors=5, weights='distance')
    cross_score[p] = numpy.mean(cross_validation.cross_val_score(
        KNeighborsRegressor, preprocessing_data, data.target,
        cv=cr_valid, scoring='mean_squared_error'))

sorted_cross_score = sorted(cross_score.items(), key=operator.itemgetter(1), reverse=True)
print("best n and cross_score", sorted_cross_score[0])
print("best n and cross_score", sorted_cross_score[-1])

