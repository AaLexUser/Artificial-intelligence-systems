import numpy as np
import pandas as pd


class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=1000, sigma=0.0000000001, optimizer='gd'):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.sigma = sigma
        self.optimizer = optimizer

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        if (self.optimizer == 'gd'):
            for _ in range(self.n_iters):
                linear_model = np.dot(X, self.weights) + self.bias
                y_predicted = self._sigmoid(linear_model)
                self.gradient_descent(X, n_samples, y, y_predicted)
        if (self.optimizer == 'newton'):
            self.newtons_method(X, y)

    def gradient_descent(self, X, n_samples, y, predictions):
        dw, db = self.grad(X, n_samples, y, predictions)
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def grad(self, X, n_samples, y, predictions):
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)
        return dw, db

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_linear(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def log_likelihood(self, X, y):
        sigmoid_probs = self._sigmoid_linear(X)
        return -np.sum(y * np.log(sigmoid_probs)
                      + (1 - y) * np.log(1 - sigmoid_probs))

    # g = X.T * (sigmoid_probs - y)
    def gradient(self, X, y, n_samples):
        sigmoid_probs = self._sigmoid_linear(X)
        return (1 / n_samples) * np.dot(X.T, (sigmoid_probs - y))

    # H = X.T * D * X
    # D = diag(sigmoid_probs * (1 - sigmoid_probs))
    def hessian(self, X, n_samples):
        sigmoid_probs = self._sigmoid_linear(X)
        print("sigmoid_probs-shape", sigmoid_probs.shape)
        H = (1 / n_samples) * np.dot(X.T, np.dot(np.diag(sigmoid_probs * (1 - sigmoid_probs)), X))
        return H

    def newtons_method(self, X, y):
        n_samples, n_features = X.shape

        # add bias to X
        X = np.column_stack((np.ones(n_samples), X))

        # n_features + 1 because of bias
        self.weights = np.zeros(n_features + 1)
        self.bias = 0


        i = 0
        dl = np.Infinity
        l = self.log_likelihood(X, y)
        while i < self.n_iters:
            i += 1
            g = self.gradient(X, y, n_samples)
            hess = self.hessian(X, n_samples)
            H_inv = np.linalg.inv(hess)
            delta = np.dot(H_inv, g)
            self.weights -= delta

            l_new = self.log_likelihood(X, y)
            dl = l_new - l
            l = l_new
            if abs(dl) < self.sigma:
                break

        # return everything as it was before)
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        return self.weights, self.bias

    def _loss(self, y, y_predicted):
        y_predicted = np.clip(y_predicted, 1e-10, 1 - 1e-10)
        return - np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))