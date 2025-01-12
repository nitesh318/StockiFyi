import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from data import generative_data


# Prophet Forecasting
def forecast_with_prophet(start_date, end_date, periods):
    data = generative_data(start_date, end_date)
    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    # Instantiate and fit the model
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods)  # Forecast for future periods
    forecast = model.predict(future)

    # Ensure forecast is a DataFrame and extract relevant columns
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper',
                            'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms',
                            'multiplicative_terms_lower', 'multiplicative_terms_upper']]

    # Filter to show only future dates
    forecast_df = forecast_df[forecast_df['ds'] > pd.to_datetime(end_date)]

    # Plotting the forecast
    fig = model.plot(forecast)
    st.pyplot(fig)

    return forecast_df


# ARIMA Forecasting Function
def forecast_with_arima(start_date, end_date, periods):
    data = generative_data(start_date, end_date)
    df_train = data.set_index('Date')["Close"]

    # Fit the ARIMA model
    model = ARIMA(df_train, order=(5, 1, 0))  # ARIMA(p, d, q)
    model_fit = model.fit()

    # Forecast future periods
    future_dates = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast = model_fit.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()

    # Prepare DataFrame to mirror Prophet output
    forecast_df = forecast_df.rename(columns={
        'mean': 'yhat',
        'mean_ci_lower': 'yhat_lower',
        'mean_ci_upper': 'yhat_upper'
    })
    forecast_df['ds'] = future_dates

    # Filter to show only future dates
    forecast_df = forecast_df[forecast_df['ds'] > pd.to_datetime(end_date)]

    # Compute Metrics
    actuals = data.set_index('Date')["Close"].tail(periods)
    predictions = forecast_df['yhat'][:len(actuals)]
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    # Display Metrics
    st.write("**Forecast Metrics:**")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plotting the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df_train, label='Actual')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='k', alpha=0.1,
                     label='Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('ARIMA Forecast')
    st.pyplot(plt)

    return forecast_df
