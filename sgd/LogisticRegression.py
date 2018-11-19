# coding: utf-8
import time
import numpy as np
import pandas as pd
import pprint
import math
import types
import matplotlib.pyplot as plt
import pprint
import math

def sigmoid(t,x):
    ''' return 0~1 '''
    index = -1 * np.dot(t, x)
    h = 1.0 / (1.0 + np.exp(index))
    return h


def logsticRegressionPredict(t,x):
    ''' return 0 or 1
    >>> logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
    1
    >>> logsticRegressionPredict( np.array([1, 3]), np.array([2, 1]))
    1
    ''' 
    h = sigmoid(t,x)
    if h >= 0.5:
        return 1
    else:
        return 0


class Network:
    '''
        Network :
            Logistic Regression
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
        self.learning_late = 0.5
        self.weight = np.zeros(2) # (1,) # np.ones((3,3)) ---> 3x3
        self.pre_cost = 0.01


    def getDataset(self):
        ''' Get american house data. '''
        df = pd.read_csv('./housing.data', header=None, sep='\s+')
        df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        self.df = df


    def devideDataset(self):
        #train_tmp = self.df[['CRIM']].values

        tmp = self.df[['RM']].values
        # panda.DataFrame ---> numpy.ndarray
        self.X_train = tmp[:450]
        self.X_test  = tmp[451:]
        self.Y_train = self.df.iloc[:450,13].values
        self.Y_test  = self.df.iloc[451:,13].values

        #print(self.X_train.shape) # 1x450
        #print(self.X_test.shape) # 1x55
        #print(self.Y_train.shape) # 1x450
        #print(self.Y_test.shape) # 1x55


    def train(self):
        # Integrate the expansion of probability--->
        integration = 0
        for ite in range(self.X_train.shape[0]):
            j = np.hstack(([1], self.X_train[ite]))
            #print(j)
            #print(j.shape)
            integration += math.log(sigmoid(self.weight, j))

        '''
        # Cost function--->
        cost = -1 * integration / self.X_train.shape[0]

        # Cost grad--->
        cost_grad = cost - self.pre_cost
        self.pre_cost = cost
        '''

        # Cost function--->
        cost = -1 * integration / self.X_train.shape[0]

        # Cost grad--->
        print(self.Y_train)
        for j in range(2):
            cost_grad = 0
            for i in range(self.X_train.shape[0]):
                f = sigmoid(self.weight, np.hstack(([1], self.X_train[i])) )
                cost_grad += (f - self.Y_train[i]) * self.X_train[i]

            # Parameter update--->
            self.weight[j] += -1 * self.learning_late * cost_grad

        #self.weight[1] += -1 * self.learning_late * cost_grad[1]
        #self.weight += -1 * self.learning_late * cost_grad
        print("weight : ",self.weight)


    def predict(self, _input):
        # t is multiple feature
        # t1*x1 + t2*x2 + t3*x3 + ... + tn*xn
        j = np.hstack(([1], _input))
        ans = sigmoid(self.weight.T, j)
        return ans


    def plotTest(self):
        plt.scatter(self.df['RM'], self.df['MEDV'])
        plt.autoscale()
        plt.grid()
        plt.show()


    def plotResult(self):
        # All data plot
        plt.scatter(self.df['RM'], self.df['MEDV'], c='red')

        # Test data plot
        plt.scatter(self.X_test, self.Y_test, c='green')

        # Predict plot
        for i in range(self.Y_test.shape[0]):
            #print(self.X_test[i].shape)
            #print(self.X_test[i].shape)
            x = self.X_test[i]
            #y = self.weight[0]*1 + self.weight[1]*self.X_test[i]
            y = np.dot(self.weight, np.hstack(([1], self.X_test[i])) )
            plt.scatter(x, y, c='blue')
            #print("x",x,"y",y)

        #x = np.linspace(0,10,4)
        #y = x
        #plt.plot(x, y, "r-")

        plt.autoscale()
        plt.grid()
        plt.show()


# main --->
network = Network()
for i in range(10):
    print("<<<",i,"loop>>>")
    network.train()
    time.sleep(0.03)
network.plotResult()
#network.calcError()
#print(sigmoid( np.array([1, 3]), np.array([2, 1])))
#print(type(df)) # <class 'pandas.core.frame.DataFrame'>


