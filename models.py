import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
# Prophet Forecasting
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

from data import generative_data_model


def forecast_with_prophet(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods)  # Forecast for future periods
    forecast = model.predict(future)

    expected_columns = {
        'ds': 'Forecast Date',
        'yhat': 'Predicted Close Price',
        'yhat_lower': 'Lower Bound (Predicted Close Price)',
        'yhat_upper': 'Upper Bound (Predicted Close Price)',
        'trend': 'Trend Component',
        'trend_lower': 'Lower Bound (Trend Component)',
        'trend_upper': 'Upper Bound (Trend Component)',
        'yearly': 'Yearly Component',
        'yearly_lower': 'Lower Bound (Yearly Component)',
        'yearly_upper': 'Upper Bound (Yearly Component)'
    }

    columns_to_include = list(expected_columns.keys())
    columns_to_include = [col for col in columns_to_include if col in forecast.columns]
    forecast_df = forecast[columns_to_include]
    forecast_df = forecast_df.rename(columns={col: expected_columns[col] for col in columns_to_include})

    if 'Forecast Date' in forecast_df.columns:
        forecast_df = forecast_df[forecast_df['Forecast Date'] > pd.to_datetime(end_date)]

    fig = model.plot(forecast)
    st.pyplot(fig)
    return forecast_df


# ARIMA Forecasting Function
def forecast_with_arima(stock_name, start_date, end_date, periods):
    data = generative_data_model(stock_name, start_date, end_date)
    df_train = data.set_index('Date')["Close"]
    model = ARIMA(df_train, order=(5, 1, 0))  # ARIMA(p, d, q)
    model_fit = model.fit()

    future_dates = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=periods)
    forecast = model_fit.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()

    expected_columns = {
        'mean': 'Predicted Close Price',
        'mean_ci_lower': 'Lower Bound (Predicted Close Price)',
        'mean_ci_upper': 'Upper Bound (Predicted Close Price)',
        'obs_ci_lower': 'Lower Bound (Observed)',
        'obs_ci_upper': 'Upper Bound (Observed)',
        'mean_se': 'Standard Error (Predicted Close Price)',
        'mean_ci_lower': 'Lower Bound (Predicted Close Price)',
        'mean_ci_upper': 'Upper Bound (Predicted Close Price)'
    }

    columns_to_include = list(expected_columns.keys())
    columns_to_include = [col for col in columns_to_include if col in forecast_df.columns]
    forecast_df = forecast_df[columns_to_include]
    forecast_df = forecast_df.rename(columns={col: expected_columns[col] for col in columns_to_include})
    forecast_df['Forecast Date'] = future_dates
    forecast_df = forecast_df[forecast_df['Forecast Date'] > pd.to_datetime(end_date)]
    actuals = data.set_index('Date')["Close"].tail(periods)
    predictions = forecast_df['Predicted Close Price'][:len(actuals)]
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Display Metrics
    st.write("**Forecast Metrics:**")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(df_train, label='Actual')
    plt.plot(forecast_df['Forecast Date'], forecast_df['Predicted Close Price'], label='Forecast')
    plt.fill_between(forecast_df['Forecast Date'], forecast_df['Lower Bound (Predicted Close Price)'],
                     forecast_df['Upper Bound (Predicted Close Price)'], color='k', alpha=0.1,
                     label='Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('ARIMA Forecast')
    st.pyplot(plt)
    return forecast_df
