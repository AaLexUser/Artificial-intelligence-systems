import numpy as np
from math import floor


def confusion_matrix_prob(pred_probs, true_y, threshold):
    res = np.zeros((2, 2))

    for pred_prob, true in zip(pred_probs, true_y):
        pred = 1 if pred_prob >= threshold else 0
        true = 1 if true == 1 else 0
        res[pred][true] += 1

    return res


def accuracy(conf):
    return (conf[1][1] + conf[0][0]) / (conf[1][1] + conf[0][0] + conf[1][0] + conf[0][1])


def recall(conf):
    return conf[1][1] / (conf[1][1] + conf[0][1])


def precision(conf):
    return conf[1][1] / (conf[1][1] + conf[1][0])


def tpr(conf):
    return recall(conf)


def fpr(conf):
    return conf[1][0] / (conf[1][0] + conf[0][0])
