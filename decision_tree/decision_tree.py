# coding: utf-8
# writer: okano

from node import _Node

import sys
import types
import numpy as np
from sklearn.base import ClassifierMixin

class DecisionTree(ClassifierMixin):
    def __init__(self, max_depth=None):
        if max_depth == None:
            version = sys.version_info[0] # <--- This meanning is major version
            if version == 2:
                self._max_depth = sys.maxint
            elif version == 3:
                self._max_depth = sys.maxsize
            else:
                raise("version error occured!!")
        else:
            self._max_depth = max_depth
        self._root = None

    def create_tree(self, X, y):
        '''
        トレーニングデータ X と トレーニング正解 y を渡すことで決定木を作成する
        '''
        self._classes = np.unique(y)
        self._root = self._grow(X, y, 0)


    def _is_leaf(self, X, y, depth):
        '''
        渡されたデータに対して、これ以上分けられるかを判定する
        条件 データが1個しかない または すべての目的変数が同じ値　または すべての説明変数が同じ値
        '''
        if len(y) == 1: # Data count is one
            return True
        elif all(y == y[0]):# or all(X == X[0]):
            return True
        elif (X == X[0]).all(): # サンプルが全部同じ特徴量をもつ場合は分岐不可能なので葉ノードを返して終了
            return True
        elif depth > self._max_depth:
            return True
        else:
            return False


    def _grow(self, X, y, depth):
        '''
        決定木を成長させる(Nodeを作る)
        現在のデータをrightとleftのNodeに分けて、再帰的に呼び出す
        どの説明変数とどの値で分けたかの情報をNodeにもたせる
        '''
        """(深さ優先で) 再帰的に木を成長させる。メモリ効率は知らん"""
        # Node create --->
        uniques, counts = np.unique(y, return_counts=True)  # 各クラスの出現回数を数える (歯抜けあり)
        counter = dict(zip(uniques, counts))
        class_count = [counter[c] if c in counter else 0 for c in self._classes]  # 各クラスの出現回数 (歯抜けなし)
        #print("class_count", class_count)
        node = _Node(class_count) # <--- Write probability to node

        if self._is_leaf(X, y, depth):
            # Recursion stop in this branch --->
            return node
        else:
            # Branch **recursion** --->
            left_X, left_y, right_X, right_y, feature_id, threshold = self._branch(X, y)
            node.feature_id = feature_id
            node.threshold = threshold
            node.left = self._grow(left_X, left_y, depth+1)
            node.right = self._grow(right_X, right_y, depth+1)
            return node

    def _calc_impurity_gini(self, right, left):
        '''
        left と right の Node を受取りジニ係数を計算する

        ある分割規則によるジニ係数の減少量を計算する (小さいほうがえらい)
        left は左側に流れるデータのサンプル (e.g.: [1,1,1,2])
        right は右側に流れるデータのサンプル (e.g.: [0,0,0,2,2])
        '''
        r_num = len(right)
        l_num = len(left)
        rl_num = r_num + l_num
        
        # Right --->
        _, counts = np.unique(right, return_counts=True)
        r_geni = 1 - (counts/r_num**2).sum() - (counts/r_num**2).sum()
        r_weight = r_num/rl_num
        # Left  --->
        _, counts = np.unique(left, return_counts=True)
        l_geni = 1 - (counts/l_num**2).sum() - (counts/l_num**2).sum()
        l_weight = l_num/rl_num

        impurity = r_weight*r_geni + l_weight*l_geni
        return impurity


    def _branching_thresholds(self, data_col):
        '''
        ブランチを分ける際に使用する閾値の候補を計算する
        '''
        unique_data = np.unique(data_col)
        return (unique_data[1:] + unique_data[:-1]) / 2


    def _branch(self, X, y):
        '''
        情報利得を計算して最適な分け方の Node を返してくれる
        '''
        #best_future = None
        #best_thre   = 0
        #best_threshold = 0

        impurity_indexes = []
        future_ids = []
        thresholds = []

        for id_, data_col in enumerate(X.T):
            thre_list = self._branching_thresholds(data_col)
            # data split by thre_list --->
            for th in thre_list:
                #left_X, left_y, right_X, right_y = \
                right_y, left_y = \
                        y[th<data_col],\
                        y[~(th<data_col)]
                # calc splitted data --->
                impurity = self._calc_impurity_gini(right_y, left_y)
                impurity_indexes.append(impurity)
                thresholds.append(th)
                future_ids.append(id_)

        # Select best threshold by most small impurity --->
        best_index = np.argmin(impurity_indexes)
        best_future_id = future_ids[best_index]
        best_threshold = thresholds[best_index]
        
        is_splits = X[:, best_future_id] < best_threshold
        return X[is_splits], y[is_splits], X[~is_splits], y[~is_splits], best_future_id, best_threshold
        #return left_X, left_y, right_X, right_y, best_future, best_threshold


    def predict(self, X):
        '''
        テストデータを決定木を使って分類する
        '''
        probas = []
        #predicted = []
        # Loop --->
        for data_row in X:
            node = self._root
            while not node.is_leaf:
                is_right = data_row[node.feature_id] > node.threshold
                if is_right:
                    node = node.right
                else:
                    node = node.left
                #print("predi count", node.data) # [14 1 1]
            class_count = node.data
            #proba = np.array(class_count)/sum(class_count)
            proba = class_count/sum(class_count)
            #print("type ", type(proba))
            #print("proba", proba) # 0 0 1
            probas.append(proba)

        # Secelt most great probability --->
        index_list = np.argmax(probas, axis=1)
        return index_list


