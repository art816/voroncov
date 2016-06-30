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
import numpy
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
#    newsgroups = datasets.fetch_20newsgroups(
#        subset='all', categories=['alt.atheism', 'sci.space'])
#    with open('data_file','wb') as output:
#         pickle.dump(newsgroups, output)
    with open('data_file', 'rb') as data:
        newsgroups = pickle.load(data)

    Vectorizer = feature_extraction.text.TfidfVectorizer()
    vectorizer_data = Vectorizer.fit_transform(newsgroups.data)

    cr_valid = cross_validation.KFold(n=numpy.size(vectorizer_data, 0),
                                  n_folds=5, random_state=241, shuffle=True)
    cross_score = OrderedDict()
    for n in range(-5, 6):
        print("n = ", n)
        clf = svm.SVC(C = 10**n, random_state=241, kernel='linear')
        cross_score[n] = numpy.mean(cross_validation.cross_val_score(
            clf, vectorizer_data, newsgroups.target,
            cv=cr_valid, scoring='accuracy'))

    sorted_cross_score = sorted(cross_score.items(),
                                key=operator.itemgetter(1), reverse=True)
    print("best n and cross_score", sorted_cross_score[0])
    clf = svm.SVC(C = 10**sorted_cross_score[0][0], 
                  random_state=241, kernel='linear')
    clf.fit(vectorizer_data, newsgroups.target)
    par_names_value = list(zip(Vectorizer.get_feature_names(), abs(clf.coef_.toarray()[0])))
    sorted_par = sorted(par_names_value,
                                key=operator.itemgetter(1), reverse=True)
    best_par = [best[0] for best in sorted_par[:10]]
    best_par.sort()
    with open('ans1_task2.txt', 'w') as file:
        for best in best_par:
            file.write(best + ',')

    #grid = {'C': [10**n for n in range(-5, 6)]}
    #cr_valid = cross_validation.KFold(n=numpy.size(vectorizer_data, 0),
    #                              n_folds=5, random_state=241, shuffle=True)
    #clf = svm.SVC(random_state=241, kernel='linear')
    #Gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cr_valid)
    #Gs.fit(vectorizer_data, newsgroups.target)
    #plot_result((clf,), data)

