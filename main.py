from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from streamlit_option_menu import option_menu

from data import generative_data, generative_data_model
from models import forecast_with_prophet, forecast_with_arima, compare_models
from services import (
    fetch_stock_info,
    get_most_active_stocks,
    get_upcoming_ipos,
    get_recent_announcements,
    get_historical_stats,
    get_stock_comparison_data,
)

st.set_page_config(layout="wide", page_title="StockiFyi", page_icon="📈")

st.sidebar.markdown(
    """
    <h1 style='text-align: center; font-size: 30px; color: #1D3557;'>
        <b>Stocki</b><b style='color: #F1A94D'>fyi</b>
    </h1>
    <h2 style='text-align: center; font-size: 24px; color: blue;'>
        <b>Customize Your Stock Filter</b>
    </h2>
    """,
    unsafe_allow_html=True,
)

start_date = st.sidebar.date_input("Start date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End date", date.today())
exchange = st.sidebar.selectbox("Select Exchange", ["NSE", "BSE"], index=1)

most_active_stocks = get_most_active_stocks(exchange)
active_stocks = (
    [stock["ticker"] for stock in most_active_stocks] if most_active_stocks else []
)

selected_stock = st.sidebar.selectbox("Select stock for prediction", active_stocks)
years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

with st.spinner("Loading data..."):
    data = generative_data(start_date, end_date)

st.success("Data has been loaded successfully!")


def run_forecast(data, period, end_date):
    data["Date"] = pd.to_datetime(data["Date"])
    df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df_train = df_train.sort_values(by="ds")  # Ensure chronological order
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period, freq="D")  # Specify daily frequency
    forecast = model.predict(future)
    return forecast[forecast["ds"] >= pd.to_datetime(end_date)]


# Run the forecast
forecast = run_forecast(data, period, end_date)

# Tabs
selected_tab = option_menu(
    menu_title=None,
    options=[
        "Dataframes",
        "Plots",
        "Statistics",
        "Forecasting",
        "Comparison"
    ],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

# Handle Comparison Tab
selected_stocks = (
    st.sidebar.multiselect("Select stocks for comparison", active_stocks)
    if selected_tab == "Comparison"
    else None
)

# Dataframes Tab
if selected_tab == "Dataframes":
    company_name = next(
        (
            stock["company"]
            for stock in most_active_stocks
            if stock["ticker"] == selected_stock
        ),
        "Company Name Not Found",
    )

    st.markdown(f"""
        <h3 style="color:#4CAF50;">{selected_stock} - {company_name} Historical Data</h3>
    """, unsafe_allow_html=True)

    st.write(
        f"Showing historical stock data for **{company_name}** ({selected_stock}) from {start_date} to {end_date}."
    )

    new_data = data.drop(columns=["Adj Close", "Volume"])
    st.dataframe(new_data, use_container_width=True)

    st.markdown(f"### {selected_stock} Forecast Data")
    st.markdown(f"""
        <p style="color:#2196F3; font-size:16px;">
            Stock price forecast for <strong>{company_name}</strong> from {end_date} to {(end_date + pd.Timedelta(days=period)).strftime('%Y-%m-%d')}.
        </p>
    """, unsafe_allow_html=True)

    forecast_filtered = forecast.drop(
        columns=[
            "additive_terms",
            "additive_terms_lower",
            "additive_terms_upper",
            "weekly",
            "weekly_lower",
            "weekly_upper",
            "yearly",
            "yearly_lower",
            "yearly_upper",
            "multiplicative_terms",
            "multiplicative_terms_lower",
            "multiplicative_terms_upper",
        ]
    ).rename(
        columns={
            "ds": "Forecast Date",
            "yhat": "Predicted Close Price",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound",
            "trend": "Trend Component",
            "trend_lower": "Lower Bound (Trend)",
            "trend_upper": "Upper Bound (Trend)",
        }
    )

    st.dataframe(forecast_filtered, use_container_width=True)

    # Recent Announcements
    announcements = get_recent_announcements(company_name)
    if announcements:
        st.markdown(f"### Recent Announcements for {company_name}")
        for announcement in announcements:
            st.write(f"**{announcement['title']}** (Date: {announcement['date']})")
            st.markdown(f"[View Announcement PDF]({announcement['link']})")
    else:
        st.write(f"No recent announcements found for {company_name}.")

if selected_tab == "Plots":
    company_name = next(
        (
            stock["company"]
            for stock in most_active_stocks
            if stock["ticker"] == selected_stock
        ),
        "Company Name Not Found",
    )
    st.markdown(
        f"<h2><span style='color: green;'>{company_name}</span> Trend Analysis</h2>",
        unsafe_allow_html=True,
    )

    # Fetch historical data
    data = generative_data_model(selected_stock, start_date, end_date)

    # Moving Averages for Trend Analysis
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    # Forecasting (Dynamic Per Stock)
    forecast = run_forecast(data, period, end_date)

    # Candlestick Chart
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # Moving Averages
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["SMA_50"],
            mode="lines",
            name="50-day SMA",
            line=dict(color="blue", width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["SMA_200"],
            mode="lines",
            name="200-day SMA",
            line=dict(color="orange", width=1.5, dash="dot"),
        )
    )

    # Forecast Data
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="purple", width=2),
        )
    )

    # Confidence Interval (Shaded Area)
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(128,0,128,0.2)", width=1),
            name="Lower Bound",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(128,0,128,0.2)", width=1),
            name="Upper Bound",
        )
    )

    # Graph Layout Updates
    fig.update_layout(
        title=f"Stock Analysis for {company_name}",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_white",
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
    )

    st.plotly_chart(fig)

if selected_tab == "Statistics":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")

    stock_info = fetch_stock_info(company_name)

    company_profile = stock_info.get('companyProfile', {})
    peer_company_list = company_profile.get('peerCompanyList', [])

    st.markdown(f"""
        <h2 style="color: green; text-align: center;">
            {selected_stock} Statistical Analysis for <span style="color: #2196F3;">{company_name}</span>
        </h2>
        <hr style="border: 2px solid #4CAF50;">
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "📊 Key Financial Metrics of Peer Companies",
        "📉 Profit and Loss Statistics of Current Company"
    ])

    with tab1:
        st.markdown("<h3 style='color: #FF9800;'>📊 Key Financial Metrics</h3>", unsafe_allow_html=True)

        data = []

        for company in peer_company_list:
            if company.get('companyName') == company_name:
                continue

            company_metrics = {
                'Company Name': company.get('companyName', 'Unknown'),
                'P/E Ratio': company.get('priceToEarningsValueRatio', 0) or 0,
                'Market Cap': company.get('marketCap', 0) or 0,
                'P/B Ratio': company.get('priceToBookValueRatio', 0) or 0,
                'Net Profit Margin (%)': company.get('netProfitMarginPercentTrailing12Month', 0) or 0,
                'Dividend Yield (%)': company.get('dividendYieldIndicatedAnnualDividend', 0) or 0
            }
            data.append(company_metrics)

        df = pd.DataFrame(data)


        def highlight_table(s):
            return ['background-color: #f4f4f4' if i % 2 == 0 else 'background-color: #ffffff' for i in range(len(s))]


        styled_df = df.style.set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{'selector': 'th', 'props': [('font-size', '14px'),
                                                            ('background-color', '#2196F3'),
                                                            ('color', 'white'),
                                                            ('text-align', 'center')]}]) \
            .apply(highlight_table, subset=pd.IndexSlice[:, :])

        st.dataframe(styled_df, use_container_width=True)

    with tab2:
        st.markdown("<h3 style='color: #E91E63;'>📉 Profit and Loss Statistics</h3>", unsafe_allow_html=True)

        historical_stats = get_historical_stats(company_name)

        if historical_stats and "profit_loss_stats" in historical_stats and historical_stats["profit_loss_stats"]:
            profit_loss_stats = historical_stats["profit_loss_stats"]
            for metric, year_data in profit_loss_stats.items():
                profit_loss_stats[metric] = {year: (0 if pd.isna(value) else value) for year, value in
                                             year_data.items()}
            profit_loss_df = pd.DataFrame(profit_loss_stats).T
            profit_loss_df.reset_index(inplace=True)
            profit_loss_df.columns = ['Metric'] + list(profit_loss_df.columns[1:])


            def highlight_table(s):
                return ['background-color: #f4f4f4' if i % 2 == 0 else 'background-color: #ffffff' for i in
                        range(len(s))]


            styled_profit_loss_df = profit_loss_df.style.set_properties(**{'text-align': 'center'}) \
                .set_table_styles([
                {'selector': 'th', 'props': [('font-size', '14px'),
                                             ('background-color', '#E91E63'),
                                             ('color', 'white'),
                                             ('text-align', 'center')]}]) \
                .apply(highlight_table, subset=pd.IndexSlice[:, :])

            st.dataframe(styled_profit_loss_df, use_container_width=True)
        else:
            st.markdown("<p style='color: red; font-size: 16px;'>⚠️ No profit and loss statistics available.</p>",
                        unsafe_allow_html=True)

if selected_tab == "Forecasting":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")

    st.sidebar.markdown("<h3 style='color:#673AB7;'>🔮 Select Forecasting Model</h3>", unsafe_allow_html=True)
    forecast_model = st.sidebar.radio("Choose Model", ["Prophet", "Arima", "Compare Both"])

    # Banner
    st.markdown(
        f"""
        <div style='padding: 8px; border-radius: 6px; background: linear-gradient(to right, #0D47A1, #1976D2);
                    text-align: center; color: white; box-shadow: 1px 2px 8px rgba(0,0,0,0.15); width: 80%; margin: auto;'>
            <h3 style='margin: 4px 0; font-size: 20px;'>📊 {selected_stock} Forecasting ({company_name})</h3>
            <p style='margin: 2px 0; font-size: 14px;'>⏳ {end_date} → {(end_date + pd.Timedelta(days=period)).strftime('%Y-%m-%d')} | {years_to_predict} Years</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Fetch forecast data
    prophet_forecast, arima_forecast = None, None

    if forecast_model in ["Prophet", "Compare Both"]:
        prophet_forecast = forecast_with_prophet(company_name, start_date, end_date, period)

        if prophet_forecast is not None and not prophet_forecast.empty:
            if "ds" in prophet_forecast.columns:
                prophet_forecast.rename(columns={"ds": "Date"}, inplace=True)
            else:
                st.error(
                    f"⚠️ 'Date' column missing in Prophet forecast! Available columns: {', '.join(prophet_forecast.columns)}")
        else:
            st.error("⚠️ Prophet forecast data is empty or unavailable.")

    if forecast_model in ["Arima", "Compare Both"]:
        arima_forecast = forecast_with_arima(company_name, start_date, end_date, period)

        if arima_forecast is not None and not arima_forecast.empty:
            if "Forecast Date" in arima_forecast.columns:
                arima_forecast.rename(columns={"Forecast Date": "Date"}, inplace=True)
        else:
            st.error("⚠️ ARIMA forecast data is empty or unavailable.")

    # Display Prophet Results
    if forecast_model == "Prophet" and prophet_forecast is not None and not prophet_forecast.empty:
        st.markdown("<h3 style='color: #4CAF50;'>📈 Prophet Forecast Results</h3>", unsafe_allow_html=True)
        st.dataframe(prophet_forecast, use_container_width=True)

    elif forecast_model == "Arima" and arima_forecast is not None and not arima_forecast.empty:
        st.markdown("<h3 style='color: #FF9800;'>📉 ARIMA Forecast Results</h3>", unsafe_allow_html=True)
        if "Date" in arima_forecast.columns:
            arima_forecast.rename(columns={"Date": "Date"}, inplace=True)
            arima_forecast["Date"] = pd.to_datetime(arima_forecast["Date"])
        else:
            st.error(f"⚠️ 'Date' column not found! Available columns: {', '.join(arima_forecast.columns)}")
            st.stop()
        st.dataframe(arima_forecast, use_container_width=True)

        if "mean" in arima_forecast.columns:
            arima_forecast.rename(columns={"mean": "Predicted Close Price"}, inplace=True)
        else:
            st.error(
                f"⚠️ 'mean' column (Predicted Close Price) not found! Available columns: {', '.join(arima_forecast.columns)}")
            st.stop()

    elif forecast_model == "Compare Both":
        comparison_df = compare_models(company_name, start_date, end_date, period)
        st.markdown("<h3 style='color: #0288D1;'>📊 Prophet vs. ARIMA Forecast Comparison</h3>",
                    unsafe_allow_html=True)

        # Display Dataframe in UI
        st.dataframe(comparison_df, use_container_width=True)

        true_values = generative_data_model(company_name, start_date, end_date).set_index("Date")["Close"].reindex(
            comparison_df["Date"], method="ffill"
        )
        prophet_mae = mean_absolute_error(true_values, comparison_df["Prophet Prediction"])
        arima_mae = mean_absolute_error(true_values, comparison_df["ARIMA Prediction"])

        st.markdown("### 📌 Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="📉 Prophet MAE", value=f"{prophet_mae:.4f}")

        with col2:
            st.metric(label="📈 ARIMA MAE", value=f"{arima_mae:.4f}")

        st.markdown("### 📊 Conclusion")
        if prophet_mae < arima_mae:
            st.success("✅ **Prophet performed better** with a lower Mean Absolute Error (MAE).")
        elif arima_mae < prophet_mae:
            st.warning("⚠️ **ARIMA performed better** with a lower MAE.")
        else:
            st.info("📊 Both models performed similarly. Choose based on your specific needs.")

# Comparison Tab
if selected_tab == "Comparison":
    def compare_stocks(stocks_data):
        comparison_results = []

        for stock in stocks_data:
            stock_metrics = stock.get("priceTarget", {})
            recommendation_metrics = stock.get("recommendation", {})

            mean = stock_metrics.get("Mean", 0)
            high = stock_metrics.get("High", 0)
            low = stock_metrics.get("Low", 0)
            recommendation_mean = recommendation_metrics.get("Mean", 0)

            # Extract snapshot means
            price_snapshot_means = [
                snapshot["Mean"]
                for snapshot in stock.get("priceTargetSnapshots", {}).get("PriceTargetSnapshot", [])
                if snapshot.get("Mean") is not None
            ]
            recommendation_snapshot_means = [
                snapshot["Mean"]
                for snapshot in stock.get("recommendationSnapshots", {}).get("RecommendationSnapshot", [])
                if snapshot.get("Mean") is not None
            ]

            avg_price_snapshot = sum(price_snapshot_means[-4:]) / len(
                price_snapshot_means[-4:]) if price_snapshot_means else mean
            avg_recommendation_snapshot = sum(recommendation_snapshot_means[-4:]) / len(
                recommendation_snapshot_means[-4:]) if recommendation_snapshot_means else recommendation_mean

            recommendation_trend = sum(
                stat.get("NumberOfAnalysts", 0)
                for stat in recommendation_metrics.get("Statistics", {}).get("Statistic", [])
            ) if recommendation_metrics.get("Statistics") else 0

            comparison_results.append({
                "Company Name": stock["company_name"],
                "Mean Target Price": mean,
                "High Target Price": high,
                "Low Target Price": low,
                "Recommendation Mean": recommendation_mean,
                "Avg Price Target Snapshot": avg_price_snapshot,
                "Avg Recommendation Snapshot": avg_recommendation_snapshot,
                "Recommendation Trend": recommendation_trend,
            })

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df["Investment Score"] = comparison_df[
            ["Mean Target Price", "Recommendation Mean", "Recommendation Trend"]].mean(axis=1)

        best_stock = comparison_df.sort_values(by="Investment Score", ascending=False).iloc[0]
        return comparison_df, best_stock


    active_stocks = get_most_active_stocks(exchange)

    if selected_stocks:
        stocks_data = []

        with st.spinner("🔄 Fetching stock data... Please wait!"):
            for stock in selected_stocks:
                company_name = next(
                    (active_stock["company"] for active_stock in active_stocks if active_stock["ticker"] == stock),
                    "Company Name Not Found",
                )

                stock_data = get_stock_comparison_data(stock)
                if stock_data:
                    stocks_data.append({**stock_data, "company_name": company_name})

        if stocks_data:
            comparison_df, best_stock = compare_stocks(stocks_data)

            st.subheader("📈 Stock Comparison Results")
            st.dataframe(comparison_df)

            st.subheader("🏆 Best Stock to Invest in Currently:")
            st.markdown(f"""
                <h3 style='color: green;'>The best stock to invest in is {best_stock['Company Name']}</h3>
                <ul>
                    <li><b>Mean Target Price:</b> {best_stock['Mean Target Price']}</li>
                    <li><b>High Target Price:</b> {best_stock['High Target Price']}</li>
                    <li><b>Low Target Price:</b> {best_stock['Low Target Price']}</li>
                    <li><b>Recommendation Mean:</b> {best_stock['Recommendation Mean']}</li>
                    <li><b>Avg Price Target Snapshot:</b> {best_stock['Avg Price Target Snapshot']}</li>
                    <li><b>Avg Recommendation Snapshot:</b> {best_stock['Avg Recommendation Snapshot']}</li>
                    <li><b>Recommendation Trend:</b> {best_stock['Recommendation Trend']}</li>
                </ul>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No valid stock data available for comparison.")
    else:
        st.warning("🔎 Please select stocks to compare.")

st.sidebar.markdown("## Upcoming IPOs")
upcoming_ipos = get_upcoming_ipos()
if upcoming_ipos:
    for ipo in upcoming_ipos:
        st.sidebar.write(f"**{ipo['name']}** ({ipo['symbol']})")
        st.sidebar.write(f"Bid starts on: {ipo.get('bidding_start_date', 'N/A')}")
        st.sidebar.write(f"Listing Date: {ipo.get('listing_date', 'N/A')}")
        st.sidebar.markdown(f"[View Document]({ipo['document_url']})")
else:
    st.sidebar.write("No upcoming IPOs found.")
