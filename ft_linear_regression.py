import pandas as pd
import matplotlib.pyplot as plt
from train import train_model
from precision import Error_mean_squared, coef_determination


def graph_maker(data, m, b):
    plt.scatter(data.km, data.price, color='black')
    plt.plot(data.km, m * data.km + b, color='red')
    plt.xlabel('km')
    plt.ylabel('Price')
    plt.savefig('linear_regression.png')


def treat_data(data):
    km_mean, km_std = data['km'].mean(), data['km'].std()
    price_mean, price_std = data['price'].mean(), data['price'].std()

    data['km_standardized'] = (data['km'] - km_mean) / km_std
    data['price_standardized'] = (data['price'] - price_mean) / price_std

    m, b = train_model(data)
    m = m * (price_std / km_std)
    b = b * price_std + price_mean - m * km_mean
    return m, b


def main():
    ask = float(input('What is the mileage of your car ?: '))
    data = pd.read_csv('data.csv')

    m, b = treat_data(data)
    print(f'average distance error : {Error_mean_squared(data, m, b):.2f}')
    print(f'The precision of the algoritmh from 0 to 1 is : \
{coef_determination(data, m, b):.2f}')
    print(f'The estimated value of your car is : {((m * ask) + b):.2f}')
    graph_maker(data, m, b)


if __name__ == "__main__":
    main()
