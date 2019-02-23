# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
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


SIGMOID_RANGE = 34.5387 # 34.538776394910684
def sigmoid(x):
    ''' return 0~1 '''
    try:
        # オーバーフロー対策
        x = np.where(x <= -SIGMOID_RANGE, 1e-10, x)
        x = np.where(x >= SIGMOID_RANGE, 1.0 - 1e-10, x)
        return 1.0 / (1.0 + np.exp(-x))
    except FloatingPointError:
<<<<<<< HEAD
        print ("FloatingPointError")
        print (x)
=======
        print "FloatingPointError"
        print x
>>>>>>> bf5f906c7396825b002f8add5d1b8dcbbf04c913
        sys.exit()


def boxPlot(y):
    # Create box-plot
    points = (y)
    fig, ax = plt.subplots()
    bp = ax.boxplot(points)

    # Plot setting
    plt.grid()
    plt.title('Error Box plot')
    plt.xlabel('Error')
    plt.ylabel('Doller')
    plt.ylim([-50,50])
    # Draw
    plt.show()


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
        self.epoch = 1
        self.eta = 0.3
        self.hidden_weight = np.ones((12,10)) #np.zeros()
        self.output_weight = np.ones(12) #np.zeros()
        #self.hidden_weight = np.random.randn(12, 10)
        #self.output_weight = np.random.randn(12)
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
        '''
            Step 1. Predict
            Step 2. Backpropagation
            Step 3. Plot
        '''
        glaph_x, glaph_y = [], []
        for epoch in range(self.epoch):
            for i, (x, y) in enumerate( zip(self.train_x, self.train_y)):
                # Predict --->
                output_node, hidden_node = self.predict(x)
                # Backpropagation --->
                self.backpropagation(x, output_node, y, hidden_node)
                # Visualize preparation --->
                glaph_x.append(i)
                glaph_y.append(output_node - y)
        # Plot --->
        if 0:
            plt.title("Train process (Doller)")
            plt.plot(glaph_x, glaph_y)
            plt.show()


    def test(self):
        glaph_x, predict, correct, error = [], [], [], []
        for i, (x, y) in enumerate( zip(self.test_x, self.test_y)):
            # Predict --->
            output, _ = self.predict(x)
            # Visualize preparation --->
            glaph_x.append(i)
            predict.append(output)
            correct.append(y)
            error.append(output - y)
        # Plot result --->
        boxPlot(error)
        # Print result --->
        # 平均
<<<<<<< HEAD
        print ("mean :", np.mean(error))
        # 分散
        print ("var  :", np.var(error))
        # 標準偏差
        print ("std  :", np.std(error))
=======
        print "mean :", np.mean(error)
        # 分散
        print "var  :", np.var(error)
        # 標準偏差
        print "std  :", np.std(error)
>>>>>>> bf5f906c7396825b002f8add5d1b8dcbbf04c913


    def predict(self, x):
        '''
            Forward Propagation
                l_  = active(Wl + b)
                l_' = active(W'l')
        '''
        # 12x1 = 12*(10+0) (10+0)x1
        hidden_node = np.dot(self.hidden_weight, x)
        #hidden_node = relu(hidden_node)
        hidden_node = sigmoid(hidden_node)

        # 1x1 = 1*(12+0) (12+0)x1
        output_node = np.dot(self.output_weight, hidden_node)
        #output_node = sigmoid(output_node)

        # Dont use activation function --->
        #print "hidden_weight", self.hidden_weight
        #print "hidden_node", hidden_node
        #print "output_weight", self.output_weight
        #print "output_node", output_node

        return output_node, hidden_node


    def backpropagation(self, x, y, r, z):
        '''
        '''
        hidden_delta = np.zeros((12,10))
        output_delta = np.zeros(12)
        # OUTPUT WEIGHT --->
        for j in range(z.size):
            '''
            eta * (r-y) * y * (1-y) * z
            '''
            # Sigmoid --->
            #output_delta[j] = self.eta * (r-y) * y * (1-y) * z[j]
            # Raw pass --->
            output_delta[j] = self.eta * (r-y) * 1 * z[j]

        # HIDDEN WEIGHT --->
        for j in range(z.size):
            for i in range(x.size):
                '''
                eta * (r-y) * y * (1-y) * weight.2jk * (1-z) * x
                '''
                # Sigmoid --->
                #hidden_delta[j][i] = self.eta * (r-y) * y * (1-y) * output_delta[j] * z[j] * (1-z[j]) * x[i]
                # Raw pass --->
                part0 = (r-y)
                part1 = round(part0, 5) * round(output_delta[j], 5)
                part2 = z[j] * (1-z[j]) * x[i]
                hidden_delta[j][i] = self.eta * round(part1, 5) * round(part2, 5)
        #print "x:", x
        #print'y:{:.3f}'.format(y), "r:", r
        #print "z:", z
        #print ("output_delta :", output_delta)
        #print ("hidden_delta :", hidden_delta)
        #self.output_weight = self.output_weight + output_delta 
        #print "before", self.output_weight[0]
        self.output_weight += output_delta 
        self.hidden_weight += hidden_delta
        #print "after ", self.output_weight[0]


#np.seterr(over="print", under="print", invalid="print")
np.seterr(all="raise")
network = Network()
network.train()
network.test()

