# coding: utf-8
# writer: okano

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
#from urllib.request import urlretrieve

from perceptron import Perceptron

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの作成
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 各特徴量を１次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
    alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:, 1], c="", alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test set')


def plot_decision_region_iris(X, y, classifier, test_idx=None,resolution=0.02):
    plot_decision_regions(X, y, classifier=classifier, test_idx=test_idx, resolution=resolution)
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.show()

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

    X = X[:, [0, 2]] # 使う特徴量を2種類に絞る
    y = np.where(y==0, -1, 1) # y==0は-1に、それ以外は1にする

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # ----->

    net = Perceptron()
    print('Fitting')
    net.fit(X_train, y_train)
    print('Access')
    print('  score_train: {}'.format(net.score(X_train, y_train)))
    print('  score_test: {}'.format(net.score(X_test, y_test)))

    plot_decision_region_iris(X, y, net)
    #dt_predict_y_train = net.predict(X_train)
    #print('  predict_y_train: {}'.format(dt_predict_y_train))
    #print('  (actual)  : {}'.format(y_train))
    #print('  score_train: {}'.format(net.score(X_train, y_train)))
    #dt_predict_y_test = net.predict(X_test)
    #print('  predict_y_test: {}'.format(dt_predict_y_test))
    #print('  (actual)  : {}'.format(y_test))
    #print('  score_test: {}'.format(net.score(X_test, y_test)))
    
    #plot(score)

