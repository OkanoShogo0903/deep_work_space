# coding: utf-8
# writer: okano
import numpy as np
from node import _Node
from sklearn.base import ClassifierMixin

class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self._root = None

    def create_tree(self, X, Y):
        '''
        トレーニングデータ X と トレーニング正解 y を渡すことで決定木を作成する
        '''
        self._targets = np.unique(y)
        self._root = self._grow(X, y)

    def _is_leaf(self, X, y):
        '''
        渡されたデータに対して、これ以上分けられるかを判定する
        条件 データが1個しかない または すべての目的変数が同じ値　または すべての説明変数が同じ値
        '''
        pass

    def _grow(self, X, y):
        '''
        決定木を成長させる(Nodeを作る)
        現在のデータをrightとleftのNodeに分けて、再帰的に呼び出す
        どの説明変数とどの値で分けたかの情報をNodeにもたせる
        '''
        # TODO is leaf implement
        node = _Node()
        left_X, left_y, right_X, right_y, feature_id, threshold = self._branch(X, y)
        node.feature_id = feature_id
        node.threshold = threshold
        node.left = self._grow(left_X, left_y)
        node.right = self._grow(right_X, right_y)
        return node
        #n = len(X)
        #print ("n:", n)
        #for i, (x, y) in enumerate(zip(X, Y)):
        #    print(i, ":", x, y)
        #    impurity

    def _delta_gini_index(self, left, right):
        '''
        left と right の Node を受取りジニ係数を計算する

        ある分割規則によるジニ係数の減少量を計算する (小さいほうがえらい)
        left は左側に流れるデータのサンプル (e.g.: [1,1,1,2])
        right は右側に流れるデータのサンプル (e.g.: [0,0,0,2,2])
        '''
        pass

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
        impurity = 0
        best_future = None
        best_thre   = 0
        for i, data_col in enumerate(X.T):
            thre_list = self._branching_thresholds(X.T)
            # data split by thre_list
            left_X, left_y, right_X, right_y
            #   calc splitted data
            gini_index = self._delta_gini_index()
            # select best threshold by most small gini
            thre = thre_list[np.argmax(gini_index)]
            #impurity = 1 - math.pow(, 2) - math.pow(, 2)
            if thre > best_thre:
                best_future = i
                best_thre   = thre

        feature_id = best_future
        threshold  = best_thre
        return left_X, left_y, right_X, right_y, feature_id, threshold


    def _predict_one(self, data_row):
        '''
        テストデータの一つを学習済みの決定木で分類する
        '''
        pass

    def _predict_proba(self, X):
        '''
        渡されたテストデータから、それぞれの分類結果の配列を返す
        '''
        pass

    def predict(self, X):
        '''
        テストデータを決定木を使って分類する
        '''
        pass


