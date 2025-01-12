import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from data import create_dummy_data


# Prophet Forecasting
def forecast_with_prophet(start_date='2020-01-01', end_date='2022-01-01'):
    data = create_dummy_data(start_date, end_date)
    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    # Instantiate and fit the model
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=365)  # Forecast for 365 days
    forecast = model.predict(future)

    # Ensure forecast is a DataFrame and extract relevant columns
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper',
                            'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms',
                            'multiplicative_terms_lower', 'multiplicative_terms_upper']]

    # Plotting the forecast
    fig = model.plot(forecast)
    st.pyplot(fig)

    return forecast_df


# SVR Forecasting
def forecast_with_svr(start_date='2020-01-01', end_date='2022-01-01', years_to_predict=1):
    data = create_dummy_data(start_date, end_date)
    df_train = data[["Date", "Close"]]
    X = pd.to_datetime(df_train['Date']).values.reshape(-1, 1)
    y = df_train['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### SVR Model - MSE: {mse:.2f}, R2: {r2:.4f}")

    future_dates = pd.date_range(start=end_date, periods=years_to_predict * 365, freq='D')
    X_future = pd.to_datetime(future_dates).values.reshape(-1, 1)

    forecast = model.predict(X_future)

    # Plotting the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(df_train['Date'], df_train['Close'], label="Historical Data")
    plt.plot(future_dates, forecast, label="Forecast", linestyle='--')
    plt.title("SVR Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(plt)

    return pd.DataFrame({'Date': future_dates, 'Predicted Close': forecast}), mse, r2


# XGBoost Forecasting
def forecast_with_xgboost(start_date='2020-01-01', end_date='2022-01-01', years_to_predict=1):
    data = create_dummy_data(start_date, end_date)
    df_train = data[["Date", "Close"]]
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Date_ordinal'] = df_train['Date'].map(pd.Timestamp.toordinal)

    X = df_train[['Date_ordinal']]
    y = df_train['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### XGBoost Model - MSE: {mse:.2f}, R2: {r2:.4f}")

    future_dates = pd.date_range(start=end_date, periods=years_to_predict * 365, freq='D')
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    forecast = model.predict(future_dates_ordinal)

    # Plotting the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(df_train['Date'], df_train['Close'], label="Historical Data")
    plt.plot(future_dates, forecast, label="Forecast", linestyle='--')
    plt.title("XGBoost Forecast")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    st.pyplot(plt)

    return pd.DataFrame({'Date': future_dates, 'Predicted Close': forecast}), mse, r2
