class _Node:
    #def __init__(self, data):
    def __init__(self, data):
        self.data = data
        self.feature_id = None
        self.threashold = None
        self.left = None # <--- left Node
        self.right = None # <--- right Node

    @property
    def is_leaf(self):
        return self.feature_id is None
