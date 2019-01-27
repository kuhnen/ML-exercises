import numpy as np


def compute_cost(X: np.array, y: np.array, thetas: np.array):
    m = np.size(X, 0)
    y_hat = X @ thetas
    diff = y_hat - y
    MSE = np.square(diff)
    J = np.sum(MSE)/(2*m)
    return J
