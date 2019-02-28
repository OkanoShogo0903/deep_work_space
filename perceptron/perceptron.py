# coding: utf-8
# writer: okano
import numpy as np
from sklearn.base import ClassifierMixin

class Perceptron(ClassifierMixin): # object
    """
    パーセプトロンの分類器
    """
    def __init__(self, eta=0.1, epoch=10):
        self.eta = eta
        self.epoch = epoch


    def fit(self, X, y):
        """
        Args:
            X : future array (xxx, 2)
            y : target array (xxx,)
        """
        #print(len(X))
        #print(type(len(X)))

        self.w = np.zeros(1 + X.shape[1]) # weight shape is (3,)

        for _ in range(self.epoch):
            for xi, target in zip(X, y):
                # 重みの更新 w0, w1, ..., Wx
                input_layer = np.hstack([ np.array(1), xi ])
                self.w[1:] += self.eta * (target - self.predict(xi)) * xi
                self.w[0]  += self.eta * (target - self.predict(xi))

        print("Weight --->", self.w)


    def net_input(self, X):
        """
        Args:
            X : future array (xxx, 2)
        Returns:
            continuous value
        """
        #print("X", X.shape)
        #print("w", self.w.shape)
        return np.dot(X, self.w[1:]) + self.w[0]


    def predict(self, X):
        """
        Args:
            X : future array (xxx, 2)
        Returns:
            label
        """
        y_ = self.net_input(X)
        #print("y", y_.shape)
        return np.where(y_>0.0, 1, -1)

