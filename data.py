import hashlib

import numpy as np
import pandas as pd


def generative_data(start_date, end_date, freq='D'):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    prices = np.random.normal(loc=100, scale=10, size=len(date_range))  # Random prices around 100
    data = pd.DataFrame({
        'Date': date_range,
        'Open': prices * np.random.uniform(0.95, 1.05, size=len(date_range)),
        'High': prices * np.random.uniform(1.00, 1.10, size=len(date_range)),
        'Low': prices * np.random.uniform(0.90, 1.00, size=len(date_range)),
        'Close': prices,
        'Adj Close': prices * np.random.uniform(0.95, 1.05, size=len(date_range)),
        'Volume': np.random.randint(1000000, 5000000, size=len(date_range))
    })
    return data


def generative_data_Arima(stock_name, start_date, end_date, freq='D'):
    seed = int(hashlib.sha256(stock_name.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    base_price = np.random.uniform(50, 500)  # Random base price between 50 and 500
    volatility = np.random.uniform(0.01, 0.05)  # Random daily volatility
    trend = np.random.uniform(-0.05, 0.05)  # Random trend (upward or downward)

    prices = [base_price]
    for _ in range(1, len(date_range)):
        daily_change = prices[-1] * np.random.uniform(-volatility, volatility)  # Random daily price change
        trend_effect = prices[-1] * trend / len(date_range)  # Add trend effect
        prices.append(prices[-1] + daily_change + trend_effect)

    data = pd.DataFrame({
        'Date': date_range,
        'Open': [p * np.random.uniform(0.95, 1.05) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.10) for p in prices],
        'Low': [p * np.random.uniform(0.90, 1.00) for p in prices],
        'Close': prices,
        'Adj Close': [p * np.random.uniform(0.95, 1.05) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, len(date_range))  # Random volume
    })
    return data
