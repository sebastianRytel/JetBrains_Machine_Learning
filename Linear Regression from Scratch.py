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
        self.coefficient = np.array(np.array(final)[0][1])

        return {'Intercept': self.intercept, 'Coefficient': self.coefficient}

    def fit_without_intercept(self, X, y):
        matrix_t = X.T
        eq_one = matrix_t @ X
        eq_two = matrix_t @ y
        final = np.linalg.inv(eq_one) @ eq_two
        self.coefficient = np.array((final))
        return {'Intercept': self.intercept, 'Coefficient': self.coefficient}

    def fit(self, X, y):
        if self.fit_intercept:
            return self.fit_w_intercept(X, y)
        self.fit_without_intercept(X, y)

    def predict(self, X):
        X = np.matrix(X)
        return np.array(X @ self.coefficient)[0]


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
