import pandas as pd


def train_model(data):
    m = 0
    b = 0
    L = 0.01
    epochs = 3000

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)

    return m, b


def standardise_data(data):
    km_mean, km_std = data['km'].mean(), data['km'].std()
    price_mean, price_std = data['price'].mean(), data['price'].std()

    data['km_standardized'] = (data['km'] - km_mean) / km_std
    data['price_standardized'] = (data['price'] - price_mean) / price_std

    m, b = train_model(data)
    m = m * (price_std / km_std)
    b = b * price_std + price_mean - m * km_mean
    create_csv(m, b)
    return m, b


def create_csv(m, b):
    csv_data = {
        'm': [m],
        'b': [b]
    }
    df = pd.DataFrame(csv_data)
    csv_file_path = 'tetas.csv'
    df.to_csv(csv_file_path, index=False)


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


def main():
    data = pd.read_csv('data.csv')
    m, b = standardise_data(data)


if __name__ == '__main__':
    main()
