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
