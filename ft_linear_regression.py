import pandas as pd
import matplotlib.pyplot as plt
from utils import check_file


TETAS_FILE = 'tetas.csv'
DATA_FILE = 'data.csv'


def graph_maker(data, m, b):
    plt.scatter(data.km, data.price, color='black')
    plt.plot(data.km, m * data.km + b, color='red')
    plt.xlabel('km')
    plt.ylabel('Price')
    plt.savefig('linear_regression.png')


def main():
    while True:
        ask = input('What is the mileage of your car ?: ')
        if ask.isdigit():
            ask = float(ask)
            break
        else:
            print('Wrong input need digits')

    data = pd.read_csv(DATA_FILE)
    if (check_file(TETAS_FILE)):
        function_parameters = pd.read_csv(TETAS_FILE)
        m, b = function_parameters['m'][0], function_parameters['b'][0]
    else:
        m, b = 0, 0

    print(f'The estimated value of your car is : {((m * ask) + b):.2f}')
    graph_maker(data, m, b)


if __name__ == "__main__":
    main()
