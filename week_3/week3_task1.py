import pandas
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
import numpy as np
import operator
from collections import OrderedDict
import matplotlib.pyplot as plt





def plot_result(svc, X):
    # create a mesh to plot in
    x_min, x_max = X[1].min() - 1, X[1].max() + 1
    y_min, y_max = X[2].min() - 1, X[2].max() + 1
    h = min(x_max - x_min, y_max - y_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']


    for i, clf in enumerate(svc):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(1, 1, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[1], X[2], c=X[0], cmap=plt.cm.Paired)
        plt.xlabel('Sepal length {} {}'.format(x_min, x_max))
        plt.ylabel('Sepal width {} {}'.format(y_min, y_max))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
    plt.show()

if __name__ == '__main__':
    data = pandas.read_csv('svm-data.csv', header=None)
    clf = svm.SVC(C = 100000<F5>, random_state=241, kernel='linear')
    clf.fit(data[[1,2]], data[0])
    print(clf.support_,)
    plot_result((clf,), data)

