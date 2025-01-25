import uuid
from datetime import date
from time import sleep

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from streamlit_option_menu import option_menu

from data import generative_data
from models import forecast_with_prophet, forecast_with_arima
from services import fetch_stock_info, get_most_active_stocks, get_upcoming_ipos, get_recent_announcements, \
    get_historical_stats, get_stock_comparison_data

st.set_page_config(layout="wide", page_title="StockiFyi", page_icon="📈")

# Sidebar
st.sidebar.markdown(
    "<h1 style='text-align: center; font-size: 30px; color: #1D3557;'><b>Stocki</b><b style='color: #F1A94D'>fyi</b></h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<h2 style='text-align: center; font-size: 24px; color: blue;'><b>Customize Your Stock Filter</b></h2>",
    unsafe_allow_html=True
)

start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

exchange = st.sidebar.selectbox("Select Exchange", ["NSE", "BSE"], index=1)

most_active_stocks = get_most_active_stocks(exchange)
if most_active_stocks:
    active_stocks = [stock['ticker'] for stock in most_active_stocks]  # Extract stock tickers
else:
    active_stocks = []

selected_stock = st.sidebar.selectbox("Select stock for prediction", active_stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

with st.spinner("Loading data..."):
    data = generative_data(start_date, end_date)
    sleep(1)

st.success("Data has been loaded successfully!")

sleep(1)
st.empty()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)
end_date_datetime = pd.to_datetime(end_date)
forecast = forecast[forecast['ds'] >= end_date_datetime]

# Dataframes Tab
selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

if selected_tab == "Comparison":
    selected_stocks = st.sidebar.multiselect("Select stocks for comparison", active_stocks)
else:
    selected_stocks = None

if selected_tab == "Dataframes":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")

    st.markdown(f"<h2><span style='color: green;'>{selected_stock} - {company_name}</span> Historical Data</h2>",
                unsafe_allow_html=True)

    st.markdown(
        f"This section displays historical stock price data for <b style='color: orange;'>{company_name}</b> (<b style='color: #F1A94D;'>{selected_stock}</b>) from <b style='color: #F1A94D;'>{start_date.strftime('%Y-%m-%d')}</b> to <b style='color: #F1A94D;'>{end_date.strftime('%Y-%m-%d')}</b>.",
        unsafe_allow_html=True
    )

    new_data = data.drop(columns=['Adj Close', 'Volume'])

    st.dataframe(new_data, use_container_width=True)

    st.markdown(f"<h2><span style='color: green;'>{selected_stock}</span> Forecast Data</h2>", unsafe_allow_html=True)
    st.markdown(
        f"This section displays the forecasted stock price data for <b style='color: orange;'>{company_name}</b> using the Prophet model from <b style='color: #F1A94D;'>{end_date.strftime('%Y-%m-%d')}</b> to <b style='color: #F1A94D;'>{(end_date + pd.Timedelta(days=period)).strftime('%Y-%m-%d')}</b>.",
        unsafe_allow_html=True
    )

    new_forecast = forecast.drop(columns=[
        'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
        'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
        'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
        'multiplicative_terms_upper'
    ]).rename(columns={
        "ds": "Forecast Date",
        "yhat": "Predicted Close Price",
        "yhat_lower": "Lower Bound (Predicted Close Price)",
        "yhat_upper": "Upper Bound (Predicted Close Price)",
        "trend": "Trend Component",
        "trend_lower": "Lower Bound (Trend Component)",
        "trend_upper": "Upper Bound (Trend Component)"
    })

    st.dataframe(new_forecast, use_container_width=True)

    announcements = get_recent_announcements(company_name)

    if announcements:
        st.markdown(f"<h3><span style='color: blue;'>Recent Announcements for {company_name}</span></h3>",
                    unsafe_allow_html=True)
        for announcement in announcements:
            title = announcement['title']
            link = announcement['link']
            date = announcement['date']
            st.write(f"**{title}** (Date: {date})")
            st.markdown(f"[View Announcement PDF]({link})")
    else:
        st.write(f"No recent announcements found for {company_name}.")

if selected_tab == "Plots":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")
    st.markdown(f"<h2><span style='color: green;'>{company_name}</span> Forecast Plot</h2>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))

    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='green')))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines',
                             line=dict(color='green', width=0), name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines',
                             line=dict(color='green', width=0), name='Upper Bound'))

    fig.update_layout(
        title=f"Stock Price Forecast for {company_name}",
        xaxis_title='Date',
        yaxis_title='Stock Price',
        showlegend=True
    )

    st.plotly_chart(fig)

if selected_tab == "Statistics":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")

    stock_info = fetch_stock_info(company_name)

    company_profile = stock_info.get('companyProfile', {})
    peer_company_list = company_profile.get('peerCompanyList', [])

    st.markdown(
        f"<h2><span style='color: green;'>{selected_stock}</span> Statistical Analysis for {company_name}</h2>",
        unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Key Financial Metrics", "Profit and Loss Statistics"])

    with tab1:
        st.markdown("### Key Financial Metrics")

        data = []

        original_company_metrics = {
            'Company Name': company_name,
            'P/E Ratio': company_profile.get('priceToEarningsValueRatio', 'N/A'),
            'Market Cap': company_profile.get('marketCap', 'N/A'),
            'P/B Ratio': company_profile.get('priceToBookValueRatio', 'N/A'),
            'Net Profit Margin (%)': company_profile.get('netProfitMarginPercentTrailing12Month', 'N/A'),
            'Dividend Yield (%)': company_profile.get('dividendYieldIndicatedAnnualDividend', 'N/A')
        }
        data.append(original_company_metrics)

        for company in peer_company_list:
            company_metrics = {
                'Company Name': company.get('companyName', 'N/A'),
                'P/E Ratio': company.get('priceToEarningsValueRatio', 'N/A'),
                'Market Cap': company.get('marketCap', 'N/A'),
                'P/B Ratio': company.get('priceToBookValueRatio', 'N/A'),
                'Net Profit Margin (%)': company.get('netProfitMarginPercentTrailing12Month', 'N/A'),
                'Dividend Yield (%)': company.get('dividendYieldIndicatedAnnualDividend', 'N/A')
            }
            data.append(company_metrics)

        df = pd.DataFrame(data)

        st.dataframe(df)

    with tab2:
        st.markdown("### Profit and Loss Statistics")

        historical_stats = get_historical_stats(company_name)

        if historical_stats and "profit_loss_stats" in historical_stats:
            profit_loss_stats = historical_stats["profit_loss_stats"]

            profit_loss_df = pd.DataFrame(profit_loss_stats).T  # Transpose to get years as columns

            column_names = ['Metric'] + list(profit_loss_df.columns)
            profit_loss_df.reset_index(inplace=True)
            profit_loss_df.columns = column_names

            st.dataframe(profit_loss_df, use_container_width=True)
        else:
            st.write("No profit and loss statistics available.")

# Forecasting Tab
if selected_tab == "Forecasting":
    st.markdown("<h2><span style='color: green;'>Forecasting</span> Information</h2>", unsafe_allow_html=True)

    # Forecast model selection
    forecast_model = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "Arima"])

    # Display stock details and forecast period
    st.markdown(f"""
        <h3><span style='color: blue;'>Stock to Forecast: {selected_stock}</span></h3>
        <h4><span style='color: orange;'>Time Frame: {end_date} to {end_date + pd.Timedelta(days=period)} / Forecasting Period: {years_to_predict} years</span></h4>
    """, unsafe_allow_html=True)

    # Forecast based on selected model
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")
    if forecast_model == "Prophet":
        forecast = forecast_with_prophet(start_date, end_date, period)
    elif forecast_model == "Arima":
        forecast = forecast_with_arima(company_name, start_date, end_date, period)

    # Display forecast results only in the Forecasting tab
    st.write("Forecast Results: ")
    st.write(forecast)

# Comparison Tab
if selected_tab == "Comparison":
    def compare_stocks(stocks_data):
        comparison_results = []

        for stock in stocks_data:
            stock_metrics = stock.get('priceTarget') or {}
            recommendation_metrics = stock.get('recommendation') or {}

            mean = stock_metrics.get('Mean') or 0
            high = stock_metrics.get('High') or 0
            low = stock_metrics.get('Low') or 0
            recommendation_mean = recommendation_metrics.get('Mean') or 0

            # Handle snapshots carefully to avoid None values
            price_snapshot_means = [snapshot['Mean'] for snapshot in
                                    stock.get('priceTargetSnapshots', {}).get('PriceTargetSnapshot', []) if
                                    snapshot.get('Mean') is not None]
            recommendation_snapshot_means = [snapshot['Mean'] for snapshot in
                                             stock.get('recommendationSnapshots', {}).get('RecommendationSnapshot', [])
                                             if snapshot.get('Mean') is not None]

            # Calculate the average of the last 4 snapshots or use mean if empty
            avg_price_snapshot = (sum(price_snapshot_means[-4:]) / len(
                price_snapshot_means[-4:])) if price_snapshot_means else mean
            avg_recommendation_snapshot = (sum(recommendation_snapshot_means[-4:]) / len(
                recommendation_snapshot_means[-4:])) if recommendation_snapshot_means else recommendation_mean

            # Ensure valid recommendation trend
            recommendation_trend = 0
            if 'Statistics' in recommendation_metrics and recommendation_metrics['Statistics']:
                recommendation_trend = sum(stat.get('NumberOfAnalysts', 0) for stat in
                                           recommendation_metrics['Statistics'].get('Statistic', []))

            comparison_results.append({
                'Company Name': stock['company_name'],
                'Mean Target Price': mean,
                'High Target Price': high,
                'Low Target Price': low,
                'Recommendation Mean': recommendation_mean,
                'Avg Price Target Snapshot': avg_price_snapshot,
                'Avg Recommendation Snapshot': avg_recommendation_snapshot,
                'Recommendation Trend': recommendation_trend
            })

        comparison_df = pd.DataFrame(comparison_results)

        # Calculate an investment score based on the desired parameters
        comparison_df['Investment Score'] = comparison_df[
            ['Mean Target Price', 'Recommendation Mean', 'Recommendation Trend']].mean(axis=1)

        # Select the best stock based on the highest investment score
        best_stock = comparison_df.sort_values(by='Investment Score', ascending=False).iloc[0]

        return comparison_df, best_stock


    st.title("Stock Comparison Tool")

    active_stocks = get_most_active_stocks(exchange)
    if selected_stocks:
        stocks_data = []

        for stock in selected_stocks:
            company_name = next(
                (active_stock['company'] for active_stock in active_stocks if active_stock['ticker'] == stock),
                "Company Name Not Found")

            stock_data = get_stock_comparison_data(stock)
            if stock_data:
                stocks_data.append({**stock_data, 'company_name': company_name})

        if stocks_data:
            comparison_df, best_stock = compare_stocks(stocks_data)

            st.subheader("Stock Comparison Results")
            st.dataframe(comparison_df)

            st.subheader("Best Stock to Invest in Currently:")
            st.write(
                f"The best stock to invest in currently is **{best_stock['Company Name']}** based on the comparison metrics.")

            st.write(f"**Mean Target Price**: {best_stock['Mean Target Price']}")
            st.write(f"**High Target Price**: {best_stock['High Target Price']}")
            st.write(f"**Low Target Price**: {best_stock['Low Target Price']}")
            st.write(f"**Recommendation Mean**: {best_stock['Recommendation Mean']}")
            st.write(f"**Avg Price Target Snapshot**: {best_stock['Avg Price Target Snapshot']}")
            st.write(f"**Avg Recommendation Snapshot**: {best_stock['Avg Recommendation Snapshot']}")
            st.write(f"**Recommendation Trend**: {best_stock['Recommendation Trend']}")
        else:
            st.warning("No valid stock data available for comparison.")
    else:
        st.warning("Please select stocks to compare.")

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
