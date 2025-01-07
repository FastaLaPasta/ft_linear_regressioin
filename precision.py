from math import sqrt
import numpy as np
import pandas as pd
from utils import check_file


TETAS_FILE = 'tetas.csv'
DATA_FILE = 'data.csv'


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


def main():
    data = pd.read_csv(DATA_FILE)
    if (check_file(TETAS_FILE)):
        function_parameters = pd.read_csv(TETAS_FILE)
        m, b = function_parameters['m'][0], function_parameters['b'][0]
    else:
        m, b = 0, 0

    print(f'average distance error : {Error_mean_squared(data, m, b):.2f}')
    print(f'The precision of the algoritmh from 0 to 1 is : \
{coef_determination(data, m, b):.2f}')


if __name__ == '__main__':
    main()
