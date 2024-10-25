import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Nairobi Office Price Ex.csv')

office_size = df['SIZE'].values
office_price = df['PRICE'].values

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * X + c
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

m = np.random.randn()  # Slope
c = np.random.randn()  # Intercept
learning_rate = 0.0001
epochs = 10

for epoch in range(epochs):
    y_pred = m * office_size + c
    error = mean_squared_error(office_price, y_pred)
    print(f"Epoch {epoch + 1}: Mean Squared Error = {error:.2f}")
    m, c = gradient_descent(office_size, office_price, m, c, learning_rate)

plt.scatter(office_size, office_price, color='blue', label='Data Points')
plt.plot(office_size, m * office_size + c, color='red', label='Line of Best Fit')
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.title("Line of Best Fit after 10 Epochs")
plt.legend()

predicted_price = m * 100 + c
print(f"Predicted price for 100 sq. ft. office size: {predicted_price:.2f}")

plt.show()
