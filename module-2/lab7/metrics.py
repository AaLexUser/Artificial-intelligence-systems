import numpy as np
def confusion_matrix(pred_y, true_y):
    res = np.zeros((2, 2))
    for pred, true in zip(pred_y, true_y):
        pred = 1 if pred == 1 else 0
        true = 1 if true == 1 else 0
        res[pred][true] += 1
    return res


def accuracy(conf):
    return (conf[1][1] + conf[0][0]) / (conf[1][1] + conf[0][0] + conf[1][0] + conf[0][1])


def recall(conf):
    return conf[1][1] / (conf[1][1] + conf[0][1])


def precision(conf):
    return conf[1][1] / (conf[1][1] + conf[1][0])

def f1_score(conf):
    return 2 * precision(conf) * recall(conf) / (precision(conf) + recall(conf))