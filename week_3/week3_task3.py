import pandas
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import feature_extraction
from sklearn import grid_search

from sklearn import svm
import pickle
import numpy as np
import operator
from collections import OrderedDict
import matplotlib.pyplot as plt
from math import exp


def my_log(y, x, w):
    """
    """
    return 1 - 1/(1 + exp(-y*(w[0]*x[0] + w[1]*x[1])))

def my_one_mult0(y, x, w):
    """
    """
    return y*x[0]*my_log(y, x, w)

def my_one_mult1(y, x, w):
    """
    """ 
    return y*x[1]*my_log(y, x, w)



def grad(w, data):
    sum_grad0 = 0
    sum_grad1 = 0
    for n in range(data.shape[0]):
        y = data.ix[n, 0]
        x = tuple(data.ix[n,1:2])
        sum_grad0 += my_one_mult0(y, x, w) 
        sum_grad1 += my_one_mult1(y, x, w)
    return (sum_grad0, sum_grad1)

def init_new_w(w, k, l, C, grad):
    """
    """
    return (w[0] + k/l*grad[0] - k*C*w[0],
            w[1] + k/l*grad[1] - k*C*w[1])

def my_ver(w, data):
    """
    """
    ver = []
    for n in range(data.shape[0]):
        ver.append(1 / (1 + exp(-w[0]*data.ix[n, 1] - w[1]*data.ix[n, 2])))
    return ver




if __name__ == '__main__':
    k = 0.1
    C = 0
    data = pandas.read_csv('data-logistic.csv', header=None)
    l = data.shape[0]
    for C in (10,):
        for k in (0.1,)*10:
            w = np.random.rand(1, 2)[0]
            print(w)
            for num_iter in range(10**4):
                new_w = init_new_w(w, k, l, C, grad(w, data))
                diff_w = np.linalg.norm(np.array(new_w) - np.array(w))
                w = new_w
                if diff_w < 1e-5:
                    break
            print(w, num_iter, k, C)
            ver = my_ver(w, data)
            print(metrics.roc_auc_score(data[0], ver))

    
    

#    Vectorizer = feature_extraction.text.TfidfVectorizer()
#    vectorizer_data = Vectorizer.fit_transform(newsgroups.data)
#
#    cr_valid = cross_validation.KFold(n=numpy.size(vectorizer_data, 0),
#                                  n_folds=5, random_state=241, shuffle=True)
#    cross_score = OrderedDict()
#    for n in range(-5, 6):
#        print("n = ", n)
#        clf = svm.SVC(C = 10**n, random_state=241, kernel='linear')
#        cross_score[n] = numpy.mean(cross_validation.cross_val_score(
#            clf, vectorizer_data, newsgroups.target,
#            cv=cr_valid, scoring='accuracy'))
#
#    sorted_cross_score = sorted(cross_score.items(),
#                                key=operator.itemgetter(1), reverse=True)
#    print("best n and cross_score", sorted_cross_score[0])
#    clf = svm.SVC(C = 10**sorted_cross_score[0][0], 
#                  random_state=241, kernel='linear')
#    clf.fit(vectorizer_data, newsgroups.target)
#    par_names_value = list(zip(Vectorizer.get_feature_names(), abs(clf.coef_.toarray()[0])))
#    sorted_par = sorted(par_names_value,
#                                key=operator.itemgetter(1), reverse=True)
#    best_par = [best[0] for best in sorted_par[:10]]
#    best_par.sort()
#    with open('ans1_task2.txt', 'w') as file:
#        for best in best_par:
#            file.write(best + ',')

    #grid = {'C': [10**n for n in range(-5, 6)]}
    #cr_valid = cross_validation.KFold(n=numpy.size(vectorizer_data, 0),
    #                              n_folds=5, random_state=241, shuffle=True)
    #clf = svm.SVC(random_state=241, kernel='linear')
    #Gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cr_valid)
    #Gs.fit(vectorizer_data, newsgroups.target)
    #plot_result((clf,), data)

