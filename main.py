import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Settings
TICKER = "AAPL"
START = "2023-01-01"
FORECAST_DAYS = 10

# Fetch stock data
df = yf.download(TICKER, start=START)
prices = df['Close'].values

# Time array
t = np.arange(len(prices))

# Estimate growth rate (mu)
returns = np.diff(prices) / prices[:-1]
mu = np.mean(returns)

# ODE Model: dS/dt = mu * S
def model(S, mu):
    return mu * S

# Euler Method
S_pred = [prices[-1]]
dt = 1

for i in range(FORECAST_DAYS):
    next_value = S_pred[-1] + model(S_pred[-1], mu) * dt
    S_pred.append(next_value)

# Plot
plt.plot(prices, label="Actual Prices")
plt.plot(range(len(prices), len(prices) + len(S_pred)), S_pred, label="Predicted Prices")
plt.legend()
plt.title("Stock Price Prediction using ODE")
plt.show()
