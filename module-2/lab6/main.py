import pandas as pd
import numpy as np

from decisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from metrics import confusion_matrix_prob, fpr, tpr, recall, precision
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns


def file_init(filename: str):
    df = pd.read_csv(filename)
    X = df.drop(['GRADE', 'STUDENT ID'], axis=1)
    y = df['GRADE']
    y = pd.Series([1 if i >= 4 else 0 for i in y])
    cols = X.columns
    cols = np.random.choice(cols, 2, replace=False)
    X = X[cols]
    return X, y


def make_prediction(X_train, y_train, X_test):
    clf = DecisionTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def make_pred_prob(X_train, y_train, X_test):
    clf = DecisionTree()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    return y_prob


def auc_plot(y_prob):
    tpr_arr = []
    fpr_arr = []
    prob = np.sort(np.unique(y_prob[:, 1]))[::-1]
    for th in np.arange(1, 0, -0.01):
        conf = confusion_matrix_prob(y_prob[:, 1], y_test, th)
        tpr_arr.append(tpr(conf))
        fpr_arr.append(fpr(conf))
    display(pd.DataFrame({'tpr': tpr_arr, 'fpr': fpr_arr}))

    plt.plot([0] + fpr_arr + [1], [0] + tpr_arr + [1], label='ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def auc_pr_plot(y_prob, y_test):
    sns.set(font_scale=1)
    sns.set_color_codes("muted")
    plt.figure(figsize=(8, 8))
    p_arr = [1]
    r_arr = [0]
    y_prob = y_prob[:, 1]
    prob = np.sort(np.unique(y_prob))[::-1]
    dtype = [('prob', 'float'), ('test', 'float')]
    array = [(prob, test) for prob, test in zip(y_prob, y_test)]
    a = np.array(array, dtype= dtype)
    a = np.sort(a, order='prob')
    y_prob, y_test = a['prob'], a['test'] 
    for th in prob:
        conf = confusion_matrix_prob(y_prob, y_test, th)
        p_arr.append(precision(conf))
        r_arr.append(recall(conf))
    display(pd.DataFrame({'Recall': r_arr, 'Precision': p_arr}))
    p_arr.append(0)
    r_arr.append(1)
    plt.plot(r_arr, p_arr, lw=2, label='PR')
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y = file_init('students-performance-evaluation.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=1234
    )
    y_prob = make_pred_prob(X_train, y_train, X_test)
    auc_pr_plot(y_prob, y_test)
