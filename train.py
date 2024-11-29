import pandas as pd


def train_model(data):
    m = 0
    b = 0
    L = 0.01
    epochs = 3000

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)

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
