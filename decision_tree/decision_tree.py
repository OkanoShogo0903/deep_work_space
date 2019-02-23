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
        $B%H%l!<%K%s%0%G!<%?(B X $B$H(B $B%H%l!<%K%s%0@52r(B y $B$rEO$9$3$H$G7hDjLZ$r:n@.$9$k(B
        '''
        self._classes = np.unique(y)
        self._root = self._grow(X, y, 0)


    def _is_leaf(self, X, y, depth):
        '''
        $BEO$5$l$?%G!<%?$KBP$7$F!"$3$l0J>eJ,$1$i$l$k$+$rH=Dj$9$k(B
        $B>r7o(B $B%G!<%?$,(B1$B8D$7$+$J$$(B $B$^$?$O(B $B$9$Y$F$NL\E*JQ?t$,F1$8CM!!$^$?$O(B $B$9$Y$F$N@bL@JQ?t$,F1$8CM(B
        '''
        if len(y) == 1: # Data count is one
            return True
        elif all(y == y[0]):# or all(X == X[0]):
            return True
        elif (X == X[0]).all(): # $B%5%s%W%k$,A4ItF1$8FCD'NL$r$b$D>l9g$OJ,4tIT2DG=$J$N$GMU%N!<%I$rJV$7$F=*N;(B
            return True
        elif depth > self._max_depth:
            return True
        else:
            return False


    def _grow(self, X, y, depth):
        '''
        $B7hDjLZ$r@.D9$5$;$k(B(Node$B$r:n$k(B)
        $B8=:_$N%G!<%?$r(Bright$B$H(Bleft$B$N(BNode$B$KJ,$1$F!":F5"E*$K8F$S=P$9(B
        $B$I$N@bL@JQ?t$H$I$NCM$GJ,$1$?$+$N>pJs$r(BNode$B$K$b$?$;$k(B
        '''
        """($B?<$5M%@h$G(B) $B:F5"E*$KLZ$r@.D9$5$;$k!#%a%b%j8zN($OCN$i$s(B"""
        # Node create --->
        uniques, counts = np.unique(y, return_counts=True)  # $B3F%/%i%9$N=P8=2s?t$r?t$($k(B ($B;uH4$1$"$j(B)
        counter = dict(zip(uniques, counts))
        class_count = [counter[c] if c in counter else 0 for c in self._classes]  # $B3F%/%i%9$N=P8=2s?t(B ($B;uH4$1$J$7(B)
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
        left $B$H(B right $B$N(B Node $B$r<u<h$j%8%K78?t$r7W;;$9$k(B

        $B$"$kJ,3d5,B'$K$h$k%8%K78?t$N8:>/NL$r7W;;$9$k(B ($B>.$5$$$[$&$,$($i$$(B)
        left $B$O:8B&$KN.$l$k%G!<%?$N%5%s%W%k(B (e.g.: [1,1,1,2])
        right $B$O1&B&$KN.$l$k%G!<%?$N%5%s%W%k(B (e.g.: [0,0,0,2,2])
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
        $B%V%i%s%A$rJ,$1$k:]$K;HMQ$9$kogCM$N8uJd$r7W;;$9$k(B
        '''
        unique_data = np.unique(data_col)
        return (unique_data[1:] + unique_data[:-1]) / 2


    def _branch(self, X, y):
        '''
        $B>pJsMxF@$r7W;;$7$F:GE,$JJ,$1J}$N(B Node $B$rJV$7$F$/$l$k(B
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
        $B%F%9%H%G!<%?$r7hDjLZ$r;H$C$FJ,N`$9$k(B
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


