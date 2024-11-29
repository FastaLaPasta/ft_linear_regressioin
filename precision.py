from math import sqrt
import numpy as np


#
def Error_mean_squared(points, m, b):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].km
        y = points.iloc[i].price

        total_error += (y - (m * x + b)) ** 2
    return sqrt(total_error / float(len(points)))


def coef_determination(data, m, b):
    y_actual = data['price']
    y_predicted = m * data['km'] + b
    y_mean = np.mean(y_actual)

    ss_residual = np.sum((y_actual - y_predicted) ** 2)
    ss_total = np.sum((y_actual - y_mean) ** 2)

    r_squared = 1 - (ss_residual / ss_total)
    return r_squared
