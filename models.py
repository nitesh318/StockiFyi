import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


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
        noise = np.random.normal(0, 1)  # Added slight noise for variation
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


def forecast_with_prophet(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)

    if data.empty:
        st.error("⚠️ No stock data available for the given time range!")
        return None

    data = data.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["ds"], y=data["y"], mode="lines", name="Actual Prices", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet Prediction",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound",
                             line=dict(color="rgba(0,0,255,0.3)"), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound",
                             line=dict(color="rgba(0,0,255,0.3)"), fill="tonexty", showlegend=False))

    fig.update_layout(title="📊 Prophet Stock Forecast", xaxis_title="Date", yaxis_title="Stock Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return forecast


def forecast_with_arima(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    df_train = data.set_index('Date')["Close"]
    model = ARIMA(df_train, order=(5, 1, 0))
    model_fit = model.fit()

    future_dates = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast = model_fit.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()
    forecast_df["Date"] = future_dates

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_train.index, y=df_train, mode="lines", name="Actual Prices", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["mean"], mode="lines", name="ARIMA Prediction",
                             line=dict(color="red")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["mean_ci_upper"], mode="lines", name="Upper Bound",
                             line=dict(color="rgba(255,0,0,0.3)"), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["mean_ci_lower"], mode="lines", name="Lower Bound",
                             line=dict(color="rgba(255,0,0,0.3)"), fill="tonexty", showlegend=False))

    fig.update_layout(title="📊 ARIMA Stock Forecast", xaxis_title="Date", yaxis_title="Stock Price",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    return forecast_df


def compare_models(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_model = Prophet()
    prophet_model.fit(df_train)
    future = prophet_model.make_future_dataframe(periods=periods)
    prophet_forecast = prophet_model.predict(future)
    prophet_forecast = prophet_forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Prophet Prediction"})
    prophet_forecast["Date"] = pd.to_datetime(prophet_forecast["Date"])

    df_train_arima = data.set_index('Date')["Close"]
    arima_model = ARIMA(df_train_arima, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    forecast_arima = arima_fit.get_forecast(steps=periods)
    arima_forecast = forecast_arima.summary_frame()
    future_dates = pd.date_range(start=df_train_arima.index[-1] + pd.Timedelta(days=1), periods=periods)
    arima_forecast["Date"] = future_dates
    arima_forecast = arima_forecast.rename(columns={"mean": "ARIMA Prediction"})
    arima_forecast["Date"] = pd.to_datetime(arima_forecast["Date"])

    comparison_df = pd.merge(prophet_forecast, arima_forecast[["Date", "ARIMA Prediction"]], on="Date", how="inner")
    comparison_df.fillna(method="ffill", inplace=True)

    return comparison_df
