# coding: utf-8
# writer: okano

from decision_tree import DecisionTree

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from urllib.request import urlretrieve

def plot(y):
    plt.plot(y)

    # Plot setting
    plt.grid()
    plt.title('Accuracy transition by max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    plt.xlim([y[0][0], y[-1][0]])
    # Draw
    plt.show()

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    print('DesisionTree: ')
    score = []
    for i in range(1,20):
        #dt = DecisionTree()
        dt = DecisionTree(max_depth=i)
        dt.create_tree(X_train, y_train)

        dt_predict_y_train = dt.predict(X_train)
        #print('  predict_y_train: {}'.format(dt_predict_y_train))
        #print('  (actual)  : {}'.format(y_train))
        #print('  score_train: {}'.format(dt.score(X_train, y_train)))
        dt_predict_y_test = dt.predict(X_test)
        #print('  predict_y_test: {}'.format(dt_predict_y_test))
        #print('  (actual)  : {}'.format(y_test))
        #print('  score_test: {}'.format(dt.score(X_test, y_test)))
        score.append((i, dt.score(X_test, y_test)))
    
    plot(score)
