# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pprint
import math
import types
import pprint

'''
step1
- ライブラリのインポート
step2
- データセットの読み込み  
step3
- データ整形
step4
- 学習データとテストデータへの分割
step5
- SGD regressorの適応
step6
- 結果のプロット
step7
- 誤差のプロット
'''

def sigmoid(t,x):
    ''' return 0~1 '''
    index = -1 * np.dot(t.T, x)
    h = 1.0 / (1.0 + np.exp(index))
    return h


def logsticRegressionPredict(t,x):
    ''' return 0 or 1
    >>> logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
<<<<<<< HEAD
=======
    1
    >>> logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
>>>>>>> 62d91d7ce5ac4ef24d7aea8e0df4bcd87c62dd4c
    1
    >>> logsticRegressionPredict( -1)
    0
    ''' 
    h = sigmoid(t,x)
    if h >= 0.5:
        return 1
    else:
        return 0


class Network:
    '''
        Network :
            Stochastic Gradient Descent(SGD)
        Data :
            505
        Variables :
            CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
    '''
    def __init__(self):
        # Dataset.
        self.getDataset()
        self.devideDataset()

        # Parameter init.
        self.eta = 0.01
        self.weight = np.zeros(505)
        self.b = 0


    def getDataset(self):
        ''' Get american house data. '''
        df = pd.read_csv('./housing.data', header=None, sep='\s+')
        df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.df = df


    def devideDataset(self):
        #train_tmp = self.df[['CRIM']].values
        #print(type(train_tmp))
        #print(train_tmp)
        train = self.df.iloc[:450, :]
        test = self.df.iloc[451:, :]

        # panda.DataFrame ---> numpy.ndarray
        self.train_data = train.values
        self.test_data = test.values

        #print(self.train_data)
        #print(self.test_data)


    def callStochasticGradientDescent(self):
        x = self.
        pass

    def predict(self):
        # t is multiple feature
        # t1*x1 + t2*x2 + t3*x3 + ... + tn*xn
        ans = transpose(theta) * x


# main
<<<<<<< HEAD
    # データセットの読み込み  
    # データ整形
    df = pd.read_csv('./housing.data', header=None, sep='\s+')
    df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # 学習データとテストデータへの分割
    # Hold out法を使う(8:2に分けるやつ)
    learn_data  = 
    test_data   = 

    #network = network()
    print(sigmoid( np.array([1, 3]), np.array([2, 1])))
    print(df) # 506rows,14columns
=======

network = Network()
network.callStochasticGradientDescent()
#print(sigmoid( np.array([1, 3]), np.array([2, 1])))
#print(type(df)) # <class 'pandas.core.frame.DataFrame'>
>>>>>>> 62d91d7ce5ac4ef24d7aea8e0df4bcd87c62dd4c


