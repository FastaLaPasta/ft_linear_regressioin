import pandas as pd
import matplotlib.pyplot as plt


def main():
    ask = float(input('What is the mileage of your car ?: '))
    data = pd.read_csv('data.csv')

    km_mean, km_std = data['km'].mean(), data['km'].std()
    price_mean, price_std = data['price'].mean(), data['price'].std()

    data['km_standardized'] = (data['km'] - km_mean) / km_std
    data['price_standardized'] = (data['price'] - price_mean) / price_std

    m = 0
    b = 0
    L = 0.01
    epochs = 3000

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)

    m = m * (price_std / km_std)
    b = b * price_std + price_mean - m * km_mean
    print(m, b)
    print(f'The estimated value of your car is : {(m * ask) + b}')

    plt.scatter(data.km, data.price, color='black')
    plt.plot(data.km, m * data.km + b, color='red')
    plt.xlabel('km')
    plt.ylabel('Price')
    plt.show()


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].km_standardized
        y = points.iloc[i].price_standardized

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b


if __name__ == "__main__":
    main()
