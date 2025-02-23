import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def generative_data_model(stock_name, start_date, end_date, freq='D'):
    seed = int(hashlib.sha256((stock_name + str(np.random.randint(0, 1000))).encode('utf-8')).hexdigest(), 16) % (
            10 ** 8)
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    base_price = np.random.uniform(50, 500)
    volatility = np.random.uniform(0.01, 0.05)
    trend = np.random.uniform(-0.05, 0.05)

    prices = [base_price]
    for _ in range(1, len(date_range)):
        daily_change = prices[-1] * np.random.uniform(-volatility, volatility)
        trend_effect = prices[-1] * trend / len(date_range)
        noise = np.random.normal(0, 1)
        prices.append(prices[-1] + daily_change + trend_effect + noise)

    data = pd.DataFrame({
        'Date': date_range,
        'Open': [p * np.random.uniform(0.95, 1.05) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.10) for p in prices],
        'Low': [p * np.random.uniform(0.90, 1.00) for p in prices],
        'Close': prices,
        'Adj Close': [p * np.random.uniform(0.95, 1.05) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, len(date_range))
    })
    return data


def forecast_with_arima(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    model = ARIMA(data.set_index('Date')["Close"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=periods)

    future_dates = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "ARIMA": forecast.predicted_mean.values  # Fix: Correctly extracting ARIMA predictions
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(
        go.Scatter(x=future_dates, y=forecast_df["ARIMA"], mode="lines", name="ARIMA", line=dict(color="red")))
    fig.update_layout(title="📊 ARIMA Stock Forecast", xaxis_title="Date", yaxis_title="Stock Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    return forecast_df


def forecast_with_prophet(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    if data.empty:
        st.error("⚠️ No stock data available for the given time range!")
        return None

    model = Prophet()
    model.fit(data.rename(columns={"Date": "ds", "Close": "y"}))
    forecast = model.predict(model.make_future_dataframe(periods=periods))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(
        go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet", line=dict(color="blue")))
    fig.update_layout(title="📊 Prophet Stock Forecast", xaxis_title="Date", yaxis_title="Stock Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    return forecast.rename(columns={"ds": "Date", "yhat": "Prophet"})


def forecast_with_lstm(stock_name, start_date, end_date, periods, look_back=10):
    data = generative_data_model(stock_name, start_date, end_date)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[["Close"]])

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back])

    X, y = np.array(X), np.array(y)

    model = Sequential([LSTM(50, return_sequences=True, input_shape=(look_back, 1)), Dropout(0.2), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=8, verbose=0)

    last_inputs = scaled_data[-look_back:]
    predictions = [model.predict(last_inputs.reshape(1, look_back, 1), verbose=0)[0, 0] for _ in range(periods)]
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({"Date": future_dates, "LSTM": predictions})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode="lines", name="LSTM", line=dict(color="green")))
    fig.update_layout(title="📊 LSTM Stock Forecast", xaxis_title="Date", yaxis_title="Stock Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    return forecast_df


def compare_models(stock_name, start_date, end_date, periods):
    with st.spinner("📊 Comparing Models..."):
        try:
            data = generative_data_model(stock_name, start_date, end_date)

            prophet_forecast = forecast_with_prophet(stock_name, start_date, end_date, periods)[["Date", "Prophet"]]
            arima_forecast = forecast_with_arima(stock_name, start_date, end_date, periods)[["Date", "ARIMA"]]
            lstm_forecast = forecast_with_lstm(stock_name, start_date, end_date, periods)[["Date", "LSTM"]]

            prophet_forecast["Prophet"] += np.random.normal(0, 1, len(prophet_forecast))
            arima_forecast["ARIMA"] += np.random.normal(0, 1, len(arima_forecast))
            lstm_forecast["LSTM"] += np.random.normal(0, 1, len(lstm_forecast))

            comparison_df = prophet_forecast.merge(arima_forecast, on="Date", how="outer") \
                .merge(lstm_forecast, on="Date", how="outer")
            comparison_df.fillna(0, inplace=True)
            return comparison_df

        except Exception as e:
            st.error(f"❌ Error while comparing models: {e}")
            return None
