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
        $B%H%l!<%K%s%0%G!<%?(B X $B$H(B $B%H%l!<%K%s%0@52r(B y $B$rEO$9$3$H$G7hDjLZ$r:n@.$9$k(B
        '''
        self._targets = np.unique(y)
        self._root = self._grow(X, y)

    def _is_leaf(self, X, y):
        '''
        $BEO$5$l$?%G!<%?$KBP$7$F!"$3$l0J>eJ,$1$i$l$k$+$rH=Dj$9$k(B
        $B>r7o(B $B%G!<%?$,(B1$B8D$7$+$J$$(B $B$^$?$O(B $B$9$Y$F$NL\E*JQ?t$,F1$8CM!!$^$?$O(B $B$9$Y$F$N@bL@JQ?t$,F1$8CM(B
        '''
        pass

    def _grow(self, X, y):
        '''
        $B7hDjLZ$r@.D9$5$;$k(B(Node$B$r:n$k(B)
        $B8=:_$N%G!<%?$r(Bright$B$H(Bleft$B$N(BNode$B$KJ,$1$F!":F5"E*$K8F$S=P$9(B
        $B$I$N@bL@JQ?t$H$I$NCM$GJ,$1$?$+$N>pJs$r(BNode$B$K$b$?$;$k(B
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
        left $B$H(B right $B$N(B Node $B$r<u<h$j%8%K78?t$r7W;;$9$k(B

        $B$"$kJ,3d5,B'$K$h$k%8%K78?t$N8:>/NL$r7W;;$9$k(B ($B>.$5$$$[$&$,$($i$$(B)
        left $B$O:8B&$KN.$l$k%G!<%?$N%5%s%W%k(B (e.g.: [1,1,1,2])
        right $B$O1&B&$KN.$l$k%G!<%?$N%5%s%W%k(B (e.g.: [0,0,0,2,2])
        '''
        pass

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
        $B%F%9%H%G!<%?$N0l$D$r3X=,:Q$_$N7hDjLZ$GJ,N`$9$k(B
        '''
        pass

    def _predict_proba(self, X):
        '''
        $BEO$5$l$?%F%9%H%G!<%?$+$i!"$=$l$>$l$NJ,N`7k2L$NG[Ns$rJV$9(B
        '''
        pass

    def predict(self, X):
        '''
        $B%F%9%H%G!<%?$r7hDjLZ$r;H$C$FJ,N`$9$k(B
        '''
        pass


