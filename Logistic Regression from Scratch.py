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
