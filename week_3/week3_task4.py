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


if __name__ == '__main__':
    data = pandas.read_csv('classification.csv', header=None)
    data = data[1::]
    TP = data[(data[0]=='1') & (data[1]=='1')].shape[0]
    TN = data[(data[0]=='0') & (data[1]=='0')].shape[0]
    FP = data[(data[0]=='0') & (data[1]=='1')].shape[0]
    FN = data[(data[0]=='1') & (data[1]=='0')].shape[0]
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    sex_dict = {'0': 0, '1': 1}
    #data[0] = data[0].apply(sex_dict.get).astype(int)
    data[0] = data[0].astype(int)
    #data[1] = data[1].apply(sex_dict.get).astype(int)
    data[1] = data[1].astype(int)
    print((TP, FP, FN, TN))
    print((TP + TN)/(TP + FP + FN + TN),
          precision,
          recall,
          2*precision*recall/(precision + recall))
    
    print(metrics.accuracy_score(data[0], data[1]),
          metrics.precision_score(data[0], data[1]),
          metrics.recall_score(data[0], data[1]),
          metrics.f1_score(data[0], data[1]))
    
    
    scores_data = pandas.read_csv('scores.csv')
    columns = scores_data.columns
    roc_aus = OrderedDict()
    pr_recall = OrderedDict()
    for column in columns[1::]: 
        roc_aus[column] = metrics.roc_auc_score(scores_data[columns[0]], scores_data[column])
        pr_recall[column] = metrics.precision_recall_curve(scores_data[columns[0]], scores_data[column])
    roc_aus = OrderedDict(sorted(roc_aus.items(), key = lambda x: x[1], reverse=True))
    print(roc_aus)
    for column in columns[1::]:
        pr_recall[column + '0.7'] = []
        for num in range(len(pr_recall[column][2])):
            if pr_recall[column][1][num] >= 0.7:
                pr_recall[column + '0.7'].append(pr_recall[column][0][num])
        print(column, max(pr_recall[column + '0.7']))
