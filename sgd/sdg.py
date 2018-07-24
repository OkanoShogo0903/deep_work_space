# coding: utf-8
import numpy as np
import pandas as pd
import pprint
import math

def sigmoid(t,x):
    ''' return 0~1 '''
    index = -1 * np.dot(t.T, x)
    h = 1.0 / (1.0 + np.exp(index))
    return h


def logstic_regression_predict(t,x):
    ''' return 0 or 1
    >>> logstic_regression_predict( np.array([1, 3]), np.array([2, 1]))
    1
    >>> logstic_regression_predict( np.array([1, 3]), np.array([2, 1]))
    1
    ''' 
    h = sigmoid(t,x)
    if h >= 0.5:
        return 1
    else:
        return 0


class network:
    def __init__(self):
        #np.array =  
        pass


    def predict(self):
        # t is multiple feature
        # t1*x1 + t2*x2 + t3*x3 + ... + tn*xn
        ans = transpose(theta) * x


# main
    df = pd.read_csv('./housing.data', header=None, sep='\s+')
    df.columns = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    #network = network()
    print(sigmoid( np.array([1, 3]), np.array([2, 1])))


