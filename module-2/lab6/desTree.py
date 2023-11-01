import pandas as pd
import numpy as np
from collections import Counter
from math import ceil, sqrt, floor
from IPython.display import display
import matplotlib.pyplot as plt
from logger import logging


class Node:
    def __init__(self, feature=None, child_nodes = None, *, value=None, proba=None):
        self.feature = feature
        self.child_nodes = child_nodes
        self.value = value
        self.proba = proba

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=60, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value, leaf_proba = self._most_common_label(y)
            return Node(value=leaf_value, proba=leaf_proba)

        # find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_gain = self._best_split(X, y, feat_idxs)
        if floor(best_gain) == 0:
            leaf_value, leaf_proba = self._most_common_label(y)
            return Node(value=leaf_value, proba=leaf_proba)
        # create child notes and call _grow_tree() recursively
        child_nodes = {}
        cat_names = np.unique(X[best_feat])
        for cat in cat_names:
            child_nodes[cat] = self._grow_tree(X[X[best_feat] == cat], y[X[best_feat] == cat], depth + 1)
        return Node(best_feat, child_nodes)

    def _best_split(self, X, y, feat_idxs):

        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            info = 0
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                info += gain
            if info >= best_gain:
                best_gain = gain
                split_idx = feat_idx

        return split_idx, best_gain

    def _information_gain(self, y, X_column, split_thresh):
        # parent node entropy
        parent_entropy = self._entropy(y)

        # generate split
        # weighted average child entropy
        weight = pd.DataFrame(X_column).value_counts()[split_thresh] / len(y)
        child_entropy = self._entropy(y[X_column == split_thresh])

        # return information gain
        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        proba = [counter[0] / len(y), counter[1] / len(y)]
        value = counter.most_common(1)[0][0]
        return value, proba

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root)[0] for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_tree(x, self.root)[1] for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value, node.proba
        feature_value = x[node.feature_name]
        if feature_value not in node.child_nodes:
            return None
        return self._traverse_tree(x, node.child_nodes[feature_value])






