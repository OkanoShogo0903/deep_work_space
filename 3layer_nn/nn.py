# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pprint
import math
import types

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

def relu(x):
    y = np.maximum(0, x)
    return y


def sigmoid(t,x):
    ''' return 0~1 '''
    index = -1 * np.dot(t.T, x)
    h = 1.0 / (1.0 + np.exp(index))
    return h


def logsticRegressionPredict(t,x):
    ''' return 0 or 1
    logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
    1
    logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
    1
    logsticRegressionPredict( -1)
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
            3layer NN
        Optim:
            Stochastic Gradient Descent(SGD)
        Data :
            505
        Variables :
            CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
    '''
    def __init__(self):
        '''
            input  layer: 10
            hidden layer: 12
            output layer: 1
        '''               
        # Parameter init. --->
        self.epoch = 3
        self.eta = 0.01
        self.hidden_weight = np.zeros((12,10+1)) #np.zeros()
        self.output_weight = np.zeros(12+1) #np.zeros()
        self.x_key = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX']
        self.y_key = ['MEDV']

        # Dataset attribute. --->
        self.getDataset()
        self.devideDataset()


    def getDataset(self):
        ''' Get american house data. '''
        # データセットの読み込み  
        df = pd.read_csv('./housing.data', header=None, sep='\s+')
        df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.df = df
        #print(df) # 506rows,14columns


    def devideDataset(self):
        ''' 学習データとテストデータの分割 '''
        # TODO Hold out法を使う(8:2に分けるやつ)
        y = self.df.loc[:, "MEDV"]
        x = self.df

        # panda.DataFrame ---> numpy.ndarray
        self.train_x  = x[self.x_key].values[:450, :]
        self.test_x   = x[self.x_key].values[451:, :]
        self.train_y  = y.values[:450]
        self.test_y   = y.values[451:]

        #print(self.train_x.shape)
        #print(self.test_x.shape)
        #print(self.train_y.shape)
        #print(self.test_y.shape)


    def train(self):
        # Input --->
        for epoch in range(self.epoch):
            for x, y in zip(self.train_x, self.train_y):
                #print(x.shape, y.shape)
                predict, hidden = self.predict(x)
                self.backpropagation(x, y, predict, hidden)


    def predict(self, x):
        '''
            Forward Propagation
                l_  = active(Wl + b)
                l_' = active(W'l')
        '''
        # 12x1 = 12x(10+1) (10+1)x1
        hidden_node = np.dot(self.hidden_weight, np.append(np.array(1), x))
        hidden_node = relu(hidden_node)

        # 1x1 = 1x(12+1) (12+1)x1
        output_node = np.dot(self.output_weight, np.append(np.array(1), hidden_node))
        # Dont use activation function --->

        return output_node, hidden_node


    def backpropagation(self, x, r, y, z):
        '''
            ErrorFunction:
                **Root squared error**
                rse = np.power(y - self.predict(x), 2)/2
            ActivationFunction:
                **Sigmoid function**
            @param
                r is correct val
                y is predict val
                z is hidden layer unit's value
        '''
        # output weight  1x13
        # hidden weight 12x11
        #self.hidden_weight = np.zeros((12,10+1)) #np.zeros()
        #self.output_weight = np.zeros(12+1) #np.zeros()

        # Output-Hidden layer --->
        # 1x13 = 1x1 * 1x1 * 13x1
        output_delta += \
                self.eta * (r - y) * y*(1 - y) * z

        # Hidden-Input layer --->
        # 12x11 = 1x1 * 1x1 * 1x13 * 13x1*13x1 * 10x1
        hidden_delta = \
                self.eta * (r - y) * y*(1 - y) * self.output_weight * z(1 - z) * x

        self.output_weight += output_delta
        self.hidden_weight += hidden_delta
        

    def callStochasticGradientDescent(self):
        #x = self.
        pass


#print(sigmoid( np.array([1, 3]), np.array([2, 1])))

network = Network()
network.train()
#network.callStochasticGradientDescent()
#print(sigmoid( np.array([1, 3]), np.array([2, 1])))
#print(type(df)) # <class 'pandas.core.frame.DataFrame'>

