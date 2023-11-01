import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import LogisticRegression as lr
import metrics as m

def file_init():
    df = pd.read_csv('/Users/aleksei/ITMO/СИИ-2023/module-2/lab7/diabetes.csv')
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

def normalize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def make_prediction(X_train, y_train, X_test):
    clf = lr.LogisticRegression(optimizer='newton')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions

def make_sklearn_prediction(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions


if __name__ == '__main__':
    X, y = file_init()
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    y_pred = make_prediction(X_train, y_train, X_test)
    conf = m.confusion_matrix(y_pred, y_test)
    print(m.accuracy(conf))
    sklearn_pred = make_sklearn_prediction(X_train, y_train, X_test)
    sklearn_conf = m.confusion_matrix(sklearn_pred, y_test)
    print(m.accuracy(sklearn_conf))