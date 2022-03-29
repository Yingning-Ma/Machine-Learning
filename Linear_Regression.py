
import numpy as np
import matplotlib.pyplot as plt

def loss(a, b, x, y):
    return 0.5/len(x) * (np.square(y - a * x - b)).sum()

def optimize(a, b, x, y):
    n = len(x)
    alpha = 0.1
    da = 1 / n * ((a * x + b - y) * x).sum()
    db = 1 / n * (a * x + b - y).sum()
    a = a - alpha * da
    b = b - alpha * db
    return a, b

def linear_regression(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b

if __name__ == '__main__':

    x = [1.3854, 1.2213, 1.1009, 1.0655, 0.9503]
    y = [2.1332, 2.0162, 1.9138, 1.8621, 1.8016]

    plt.scatter(x, y)

    # y = ax + b + ε
    a, b = 0,0
    x = np.array(x)
    y = np.array(y)
    a, b = linear_regression(a, b, x, y, 100000)
    y_hat = a * x + b
    plt.plot(x, y_hat)
    plt.show()
