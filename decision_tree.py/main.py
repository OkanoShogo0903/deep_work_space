# coding: utf-8
# writer: okano

from decision_tree import DecisionTree

from sklearn import datasets
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve
import pandas as pd

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    dt = DecisionTree(max_depth=3)
    dt.create_tree(X_train, y_train)

    print('DesisionTree: ')
    dt_predict_y_train = dt.predict(X_train)
    print('  predict_y_train: {}'.format(dt_predict_y_train))
    print('  (actual)  : {}'.format(y_train))
    print('  score_train: {}'.format(dt.score(X_train, y_train)))
    dt_predict_y_test = dt.predict(X_test)
    print('  predict_y_test: {}'.format(dt_predict_y_test))
    print('  (actual)  : {}'.format(y_test))
    print('  score_test: {}'.format(dt.score(X_test, y_test)))
