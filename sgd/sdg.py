# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pprint
import math

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
    1
    >>> logsticRegressionPredict( -1)
    0
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


