============ STAGE2 ============ 

import pandas as pd
import numpy as np

class CustomLinearRegression:

    def __init__(self, *, fit_intercept):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...

    def fit_w_intercept(self, X, y):
        ones_numbers = np.array([1 for _ in range(X.shape[0])])
        X = np.matrix(X)
        matrix_w_ones = np.insert(X, 0, ones_numbers, axis=1)
        matrix_t = matrix_w_ones.T
        eq_one = matrix_t @ matrix_w_ones
        eq_two = matrix_t @ y
        final = np.linalg.inv(eq_one) @ eq_two
        self.intercept = float(np.array(final)[0][0])
        self.coefficient = np.array(final)[0][1:]

        return {'Intercept': self.intercept, 'Coefficient': self.coefficient}

    def fit_without_intercept(self, X, y):
        matrix_t = X.T
        eq_one = matrix_t @ X
        eq_two = matrix_t @ y
        final = np.linalg.inv(eq_one) @ eq_two
        self.coefficient = np.array(final)
        return {'Intercept': self.intercept, 'Coefficient': self.coefficient}

    def fit(self, X, y):
        if self.fit_intercept:
            return self.fit_w_intercept(X, y)
        self.fit_without_intercept(X, y)

    def predict(self, X):
        X = np.matrix(X)
        return np.array(X @ self.coefficient)


data = {'x': [4, 4.5, 5, 5.5, 6, 6.5, 7],
        'w': [1, -3, 2, 5, 0, 3, 6],
        'z': [11, 15, 12, 9, 18, 13, 16],
        'y': [33, 42, 45, 51, 53, 61, 62]}

# data = {'x': [1, 2, 3, 4, 10.5],
#         'w': [7.5, 10, 11.6, 7.8, 13],
#         'z': [26.7, 6.6, 11.9, 72.5, 2.1],
#         'y': [105.6, 210.5, 177.9, 154.7, 160]}

df = pd.DataFrame(data)

regCustom = CustomLinearRegression(fit_intercept=False)

result = regCustom.fit(df[['x', 'w', 'z']], df['y'])

y_pred = regCustom.predict(df[['x', 'w', 'z']])

print(y_pred)

============ STAGE3 ============

from pprint import pprint

import pandas as pd
import numpy as np

class CustomLinearRegression:

    def __init__(self, *, fit_intercept):

        self.fit_intercept = fit_intercept
        self.coefficient = ...
        self.intercept = ...
        self.r2_score_result = ...
        self.rmse_result = ...
        self.matrix_w_ones = ...

    def fit_w_intercept(self, X, y):
        ones_numbers = [1 for _ in range(X.shape[0])]
        X.insert(0, "ones", ones_numbers)
        self.matrix_w_ones = X.copy(deep=True)
        matrix_t = self.matrix_w_ones.T
        eq_one = matrix_t @ X
        eq_two = matrix_t @ y
        final = np.linalg.inv(eq_one) @ eq_two
        self.intercept = float(np.array(final)[0])
        self.coefficient = final[1:]

    def fit_without_intercept(self, X, y):
        matrix_t = X.T
        eq_one = matrix_t @ X
        eq_two = matrix_t @ y
        final = np.linalg.inv(eq_one) @ eq_two
        self.coefficient = final


    def fit(self, X, y):
        if self.fit_intercept:
            return self.fit_w_intercept(X, y)
        return self.fit_without_intercept(X, y)

    def predict(self, X):
        if self.fit_intercept:
            coefficients = np.append(self.intercept, self.coefficient)
            return self.matrix_w_ones @ coefficients
        return np.array(X @ self.coefficient)

    def r2_score(self, y, ythat):

        eq_top = sum([(yi - yithat)**2 for yi, yithat in zip(y, ythat)])
        eq_bottom = sum([(yi - np.mean(y))**2 for yi in y])

        self.r2_score_result = 1 - (eq_top / eq_bottom)
        return self.r2_score_result

    def rmse(self, y, ythat):
        eq_top = sum([(yi - yithat) ** 2 for yi, yithat in zip(y, ythat)])
        eq_semi = eq_top / len(y)
        self.rmse_result = eq_semi**0.5
        return self.rmse_result

    def final_results(self):
        return {"Intercept": self.intercept,
                "Coefficient": self.coefficient,
                "R2": self.r2_score_result,
                "RMSE": self.rmse_result}

data = {'x': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
        'w': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
        'y': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]}


df = pd.DataFrame(data)

regCustom = CustomLinearRegression(fit_intercept=True)

result = regCustom.fit(df[['x', 'w']], df['y'])

y_pred = regCustom.predict(df[['x', 'w']])

r2_score = regCustom.r2_score(df['y'], y_pred)
rmse = regCustom.rmse(df['y'], y_pred)

print(regCustom.final_results())

