========== STAGE 1 ==========

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + (np.exp(-t)))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_[1:])
        return self.sigmoid(t)

    def standardization(self, array):
        mean = sum(array) / len(array)
        mean_diff = list(map(lambda x: (x - mean) **2, array))
        std_deviation = np.sqrt(sum(mean_diff) / len(array))
        standardized_values = list(map(lambda x: (x - mean) / std_deviation, array))
        return standardized_values

data_frame = load_breast_cancer(as_frame=True)
X = data_frame['data'].loc[:, ['worst concave points', 'worst perimeter']]
y = load_breast_cancer().target

logistic = CustomLogisticRegression(fit_intercept=True)

X['worst concave points'] = logistic.standardization(X['worst concave points'])
X['worst perimeter'] = logistic.standardization(X['worst perimeter'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

coef_ = [0.77001597, -2.12842434, -2.39305793]

result = np.round(logistic.predict_proba(X_test.iloc[:10], coef_), 5).tolist()

print(result)

========== STAGE 2 ==========

import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + (np.exp(-t)))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        # self.coef_ = np.array([random.random() for _ in range(X_train.shape[1] + 1)])
        self.coef_ = [0,0,0,0]
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coef_[1:] = list(map(lambda z: (self.coef_[z+1] - self.l_rate * (y_hat - y_train[i]) *
                                                     y_hat * (1 - y_hat) * row[z]), [i for i in range(len(row))]))

    def predict(self, X_test, cut_off = 0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat >= cut_off:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def standardization(self, array):
        mean = sum(array) / len(array)
        mean_diff = list(map(lambda x: (x - mean) ** 2, array))
        std_deviation = np.sqrt(sum(mean_diff) / len(array))
        standardized_values = list(map(lambda x: (x - mean) / std_deviation, array))
        return standardized_values


data_frame = load_breast_cancer(as_frame=True)
X = data_frame['data'].loc[:, ['worst concave points', 'worst perimeter', 'worst radius']]
y = load_breast_cancer().target

logistic = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch= 1000)

X['worst concave points'] = logistic.standardization(X['worst concave points'])
X['worst perimeter'] = logistic.standardization(X['worst perimeter'])
X['worst radius'] = logistic.standardization(X['worst radius'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

training_dataset = pd.read_csv('example_stage2-3.txt')

# X_test = training_dataset.iloc[0:, :-1]
# y_train = training_dataset.iloc[0:, -1]

X_test = X_test.to_numpy()
X_train = X_train.to_numpy()

logistic.fit_mse(X_train, y_train)

y_pred = logistic.predict(X_test)

dict_ = {}

dict_["coef_"] = logistic.coef_
dict_["accuracy"] = np.round(accuracy_score(y_pred, y_test), 2)

print(dict_)

========== STAGE 3 ==========

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + (np.exp(-t)))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.coef_ = [0,0,0,0]
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coef_[1:] = list(map(lambda z: self.coef_[z+1] - self.l_rate * (y_hat - y_train[i]) *
                                                     y_hat * (1 - y_hat) * row[z], [i for i in range(len(row))]))

    def fit_log_lost(self, X_train, y_train):
        self.coef_ = [0, 0, 0, 0]
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - ((self.l_rate * (y_hat - y_train[i]))/len(X_train))

                self.coef_[1:] = list(map(lambda z: self.coef_[z + 1] - ((self.l_rate * (y_hat - y_train[i]) * row[z])
                                                                         /len(X_train)), [i for i in range(len(row))]))

    def predict(self, X_test, cut_off = 0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat >= cut_off:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def standardization(self, array):
        mean = sum(array) / len(array)
        mean_diff = list(map(lambda x: (x - mean) ** 2, array))
        std_deviation = np.sqrt(sum(mean_diff) / len(array))
        standardized_values = list(map(lambda x: (x - mean) / std_deviation, array))
        return standardized_values


data_frame = load_breast_cancer(as_frame=True)
X = data_frame['data'].loc[:, ['worst concave points', 'worst perimeter', 'worst radius']]
y = load_breast_cancer().target

logistic = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch= 1000)

X['worst concave points'] = logistic.standardization(X['worst concave points'])
X['worst perimeter'] = logistic.standardization(X['worst perimeter'])
X['worst radius'] = logistic.standardization(X['worst radius'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

X_test = X_test.to_numpy()
X_train = X_train.to_numpy()

# logistic.fit_mse(X_train, y_train)
logistic.fit_log_lost(X_train, y_train)

y_pred = logistic.predict(X_test)

dict_ = {}

dict_["coef_"] = logistic.coef_
dict_["accuracy"] = np.round(accuracy_score(y_pred, y_test), 2)

print(dict_)

========== STAGE 4 ==========

import math
from copy import copy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import numpy as np

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + (np.exp(-t)))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_[1:])
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        self.mse_first_err = []
        self.mse_last_err = []
        self.coef_ = [0,0,0,0]
        X_train = X_train.to_numpy()
        for n in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coef_[1:] = list(map(lambda z: self.coef_[z+1] - self.l_rate * (y_hat - y_train[i]) *
                                                     y_hat * (1 - y_hat) * row[z], [i for i in range(len(row))]))
                error = (y_hat - y_train[i]) ** 2
                if n == 0:
                    self.mse_first_err.append(error)
                elif n == self.n_epoch - 1:
                    self.mse_last_err.append(error)


    def fit_log_lost(self, X_train, y_train):
        self.logloss_first_err = []
        self.logloss_last_err = []
        self.coef_ = [0, 0, 0, 0]
        X_train = X_train.to_numpy()
        for n in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                self.coef_[0] = self.coef_[0] - ((self.l_rate * (y_hat - y_train[i]))/len(X_train))
                self.coef_[1:] = list(map(lambda z: self.coef_[z + 1] - ((self.l_rate * (y_hat - y_train[i]) * row[z])
                                                                         /len(X_train)), [i for i in range(len(row))]))

                error = y_train[i] * math.log(y_hat) + (1 - y_train[i]) * math.log(1-y_hat)
                if n == 0:
                    self.logloss_first_err.append(error)
                elif n == self.n_epoch - 1:
                    self.logloss_last_err.append(error)


    def predict(self, X_test, cut_off = 0.5):
        predictions = []
        X_test = X_test.to_numpy()
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            if y_hat >= cut_off:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    @classmethod
    def new_logistic_object(cls):
        return cls(fit_intercept=True, l_rate=0.01, n_epoch= 1000)


def standardization(array):
    mean = sum(array) / len(array)
    mean_diff = list(map(lambda x: (x - mean) ** 2, array))
    std_deviation = np.sqrt(sum(mean_diff) / len(array))
    standardized_values = list(map(lambda x: (x - mean) / std_deviation, array))
    return standardized_values

def standardize():
    X['worst concave points'] = standardization(X['worst concave points'])
    X['worst perimeter'] = standardization(X['worst perimeter'])
    X['worst radius'] = standardization(X['worst radius'])

data_frame = load_breast_cancer(as_frame=True)
X = data_frame['data'].loc[:, ['worst concave points', 'worst perimeter', 'worst radius']]
y = load_breast_cancer().target
standardize()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

logistic = CustomLogisticRegression.new_logistic_object()
logistic_2 = CustomLogisticRegression.new_logistic_object()
sklearn = LogisticRegression(fit_intercept=True)


logistic.fit_mse(X_train, y_train)
logistic_2.fit_log_lost(X_train, y_train)
sklearn.fit(X_train, y_train)


dict_ = {}

def results(logistic):
    y_pred = logistic.predict(X_test)
    return np.round(accuracy_score(y_pred, y_test), 2)

method_names = ["mse_accuracy" , "logloss_accuracy", "sklearn_accuracy"]
accuracy_scores = [results(logistic), results(logistic_2), results(sklearn)]

for name, scores in zip(method_names, accuracy_scores):
    dict_[name] = scores

dict_['mse_error_first'] = logistic.mse_first_err
dict_['mse_error_last'] = logistic.mse_last_err
dict_['logloss_error_first'] = logistic_2.logloss_first_err
dict_['logloss_error_last'] = logistic_2.logloss_last_err

print(dict_)

print(f"""
Answers to the questions:
1) 0.00001
2) 0.00000
3) 0.00153
4) 0.00600
5) expanded
6) expanded
""")
