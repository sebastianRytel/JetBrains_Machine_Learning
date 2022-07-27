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

