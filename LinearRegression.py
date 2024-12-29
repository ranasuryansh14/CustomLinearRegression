import numpy as np

# Custom Linear Regression Model
class MyReg:
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.coeff = np.zeros(self.n)
        self.intercept = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        del_coeff = -((self.X.T).dot(self.Y - Y_prediction)) / self.m
        del_intercept = -np.sum(self.Y - Y_prediction) / self.m
        self.coeff = self.coeff - self.learning_rate * del_coeff
        self.intercept = self.intercept - self.learning_rate * del_intercept

    def predict(self, X):
        return X.dot(self.coeff) + self.intercept

