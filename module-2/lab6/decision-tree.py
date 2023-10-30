import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from math import ceil, sqrt, floor
from IPython.display import display
import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(name)s] - %(levelname)s \n %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

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
        logger.debug("Starting _grow_tree with depth: %s, n_labels: %s, n_samples: %s", depth, n_labels, n_samples)
        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
        if round(best_gain, 5) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        # create child notes and call _grow_tree() recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        logger.debug("Starting _best_split for X \n %s \n y \n %s\n feat_indxs: %s", np.unique(X), np.unique(y),
                     feat_idxs)
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            logger.debug("feat_idx: %s, X_column: %s, thresholds: %s", feat_idx, np.unique(X_column), thresholds)
            for threshold in thresholds:

                gain = self._information_gain(y, X_column, threshold)

                if gain >= best_gain:
                    logger.debug("For thresholds %s \n "
                                 "chosen threshold %d with gain %s", thresholds, threshold, round(gain, 5))
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        logger.debug("At the end we chosen split_thresh: %d, split_idx: %s, best_gain: %s", split_thresh, split_idx,
                     round(best_gain, 5))
        return split_idx, split_thresh, best_gain

    def _information_gain(self, y, X_column, split_thresh):
        # parent node entropy
        parent_entropy = self._entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # weighted average child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # return information gain
        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


df = pd.read_csv('./students-performance-evaluation.csv')
X = df.drop(['GRADE', 'STUDENT ID'], axis=1)
y = df['GRADE']
cols = X.columns
cols = np.random.choice(cols, 2, replace=False)
X = X[cols]
logger.debug("Features we've chosen: %s", cols)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=1234
)

clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


acc = accuracy_score(y_test, predictions)
print(acc)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_features=ceil(sqrt(X_train.shape[1])))
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
print(accuracy(y_test, clf.predict(X_test)))

from sklearn.metrics import multilabel_confusion_matrix

print(multilabel_confusion_matrix(y_test, predictions))