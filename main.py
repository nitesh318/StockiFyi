import uuid
from datetime import date
from time import sleep

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from streamlit_option_menu import option_menu

from services import fetch_stock_info, get_most_active_stocks, get_upcoming_ipos, get_recent_announcements, \
    get_historical_stats, get_stock_comparison_data


# Dummy Data Generation
def create_dummy_data(start_date, end_date, freq='D'):
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


# Set page layout to wide
st.set_page_config(layout="wide", page_title="Forecastify", page_icon="📈")

# Sidebar
st.sidebar.markdown(
    "<h1 style='text-align: center; font-size: 30px;'><b>Forecasti.</b><b style='color: orange'>fy</b></h1>",
    unsafe_allow_html=True)
st.sidebar.title("Options")
start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

# Exchange selection for active stocks
exchange = st.sidebar.selectbox("Select Exchange", ["NSE", "BSE"], index=1)

# Fetch and display most active stocks based on the selected exchange
most_active_stocks = get_most_active_stocks(exchange)
if most_active_stocks:
    active_stocks = [stock['ticker'] for stock in most_active_stocks]  # Extract stock tickers
else:
    active_stocks = []

# Stock selection
selected_stock = st.sidebar.selectbox("Select stock for prediction", active_stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", active_stocks)

# IPO section in Sidebar
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

# Years of prediction slider
years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

# Display a loading spinner while loading data
with st.spinner("Loading data..."):
    data = create_dummy_data(start_date, end_date)
    sleep(1)

# Display the success message
st.success("Data loaded successfully!")

# Introduce a delay before clearing the success message
sleep(1)
st.empty()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Convert end_date to datetime
end_date_datetime = pd.to_datetime(end_date)

# Filter forecast based on end_date
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

if selected_tab == "Dataframes":
    # Find the company name from the selected stock (ticker)
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")

    # Display the header with company name and stock ticker
    st.markdown(f"<h2><span style='color: orange;'>{selected_stock} - {company_name}</span> Historical Data</h2>",
                unsafe_allow_html=True)

    # Display the description with selected stock and date range
    st.write(
        f"This section displays historical stock price data for {company_name} ({selected_stock}) from {start_date} to {end_date}.")

    # Assuming 'data' is a DataFrame, drop 'Adj Close' and 'Volume' columns
    new_data = data.drop(columns=['Adj Close', 'Volume'])

    # Display the dataframe with the cleaned data
    st.dataframe(new_data, use_container_width=True)

    # Display Forecasted Data
    st.markdown(f"<h2><span style='color: orange;'>{selected_stock}</span> Forecast Data</h2>", unsafe_allow_html=True)
    st.write(
        f"This section displays the forecasted stock price data for {selected_stock} using the Prophet model from {end_date} to {end_date + pd.Timedelta(days=period)}.")
    new_forecast = forecast.drop(columns=[
        'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
        'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
        'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
        'multiplicative_terms_upper'
    ]).rename(columns={
        "ds": "Date", "yhat": "Close", "yhat_lower": "Close Lower",
        "yhat_upper": "Close Upper", "trend": "Trend",
        "trend_lower": "Trend Lower", "trend_upper": "Trend Upper"
    })
    st.dataframe(new_forecast, use_container_width=True)

    # Fetch recent announcements for the selected stock
    announcements = get_recent_announcements(company_name)

    if announcements:
        st.markdown(f"<h3><span style='color: orange;'>Recent Announcements for {company_name}</span></h3>",
                    unsafe_allow_html=True)
        # Display the announcements
        for announcement in announcements:
            title = announcement['title']
            link = announcement['link']
            date = announcement['date']
            st.write(f"**{title}** (Date: {date})")
            st.markdown(f"[View Announcement PDF]({link})")
    else:
        st.write(f"No recent announcements found for {company_name}.")

# Plots Tab - Interactive Plot
if selected_tab == "Plots":
    company_name = next((stock['company'] for stock in most_active_stocks if stock['ticker'] == selected_stock),
                        "Company Name Not Found")
    st.markdown(f"<h2><span style='color: orange;'>{company_name}</span> Forecast Plot</h2>", unsafe_allow_html=True)

    # Interactive Plot using plotly
    fig = go.Figure()

    # Historical data plot
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))

    # Forecast data plot
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='orange')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines',
                             line=dict(color='orange', width=0), name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines',
                             line=dict(color='orange', width=0), name='Upper Bound'))

    # Add layout details
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

    # Fetch stock information for the selected company
    stock_info = fetch_stock_info(company_name)

    # Extract 'companyProfile' and 'peerCompanyList' safely
    company_profile = stock_info.get('companyProfile', {})
    peer_company_list = company_profile.get('peerCompanyList', [])

    # Display the company name and stock ticker
    st.markdown(
        f"<h2><span style='color: orange;'>{selected_stock}</span> Statistical Analysis for {company_name}</h2>",
        unsafe_allow_html=True)

    # Create tabs for organized display
    tab1, tab2 = st.tabs(["Key Financial Metrics", "Profit and Loss Statistics"])

    # Key Financial Metrics Tab
    with tab1:
        st.markdown("### Key Financial Metrics")

        # Prepare data to display in a table format
        data = []

        # Add the original company's metrics to the data list
        original_company_metrics = {
            'Company Name': company_name,
            'P/E Ratio': company_profile.get('priceToEarningsValueRatio', 'N/A'),
            'Market Cap': company_profile.get('marketCap', 'N/A'),
            'P/B Ratio': company_profile.get('priceToBookValueRatio', 'N/A'),
            'Net Profit Margin (%)': company_profile.get('netProfitMarginPercentTrailing12Month', 'N/A'),
            'Dividend Yield (%)': company_profile.get('dividendYieldIndicatedAnnualDividend', 'N/A')
        }
        data.append(original_company_metrics)

        # Add peer companies' metrics to the data list
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

        # Convert data to a DataFrame for a better table view
        df = pd.DataFrame(data)

        # Display the DataFrame in the Streamlit app
        st.dataframe(df)

    # Profit and Loss Statistics Tab
    with tab2:
        st.markdown("### Profit and Loss Statistics")

        # Fetch Historical Stats using the company name
        historical_stats = get_historical_stats(company_name)

        if historical_stats and "profit_loss_stats" in historical_stats:
            # Extract 'profit_loss_stats' from historical stats
            profit_loss_stats = historical_stats["profit_loss_stats"]

            # Create a DataFrame to display in a table
            profit_loss_df = pd.DataFrame(profit_loss_stats).T  # Transpose to get years as columns

            # Dynamically set column names based on the number of columns in the DataFrame
            column_names = ['Metric'] + list(profit_loss_df.columns)
            profit_loss_df.reset_index(inplace=True)
            profit_loss_df.columns = column_names

            # Display the data in a table
            st.dataframe(profit_loss_df, use_container_width=True)
        else:
            st.write("No profit and loss statistics available.")

# Forecasting Tab - Showcase
if selected_tab == "Forecasting":
    st.markdown("<h2><span style='color: orange;'>Forecasting</span> Information</h2>", unsafe_allow_html=True)
    st.write("This section presents the forecasting features.")

# Comparison Tab
# In the Comparison Tab, loop over selected_stocks and perform comparison for each stock

if selected_tab == "Comparison":
    def compare_stocks(stocks_data):
        comparison_results = []

        for stock in stocks_data:
            stock_metrics = stock.get('priceTarget', {})
            recommendation_metrics = stock.get('recommendation', {})

            # Extracting relevant metrics for comparison
            mean = stock_metrics.get('Mean', 0)
            high = stock_metrics.get('High', 0)
            low = stock_metrics.get('Low', 0)
            recommendation_mean = recommendation_metrics.get('Mean', 0)

            # Extracting Price Target Snapshots (for more detailed analysis)
            price_snapshot_means = [snapshot['Mean'] for snapshot in
                                    stock.get('priceTargetSnapshots', {}).get('PriceTargetSnapshot', [])]
            recommendation_snapshot_means = [snapshot['Mean'] for snapshot in
                                             stock.get('recommendationSnapshots', {}).get('RecommendationSnapshot', [])]

            # Calculating mean of the last 4 snapshots if available
            avg_price_snapshot = sum(price_snapshot_means[-4:]) / len(
                price_snapshot_means[-4:]) if price_snapshot_means else mean
            avg_recommendation_snapshot = sum(recommendation_snapshot_means[-4:]) / len(
                recommendation_snapshot_means[-4:]) if recommendation_snapshot_means else recommendation_mean

            # Additional metric: Historical recommendation trend (higher number of analysts for certain recommendations)
            recommendation_trend = sum(
                stat['NumberOfAnalysts'] for stat in recommendation_metrics.get('Statistics', {}).get('Statistic', []))

            # Adding the stock data to the comparison results
            comparison_results.append({
                'Stock ID': stock['stock_id'],
                'Mean Target Price': mean,
                'High Target Price': high,
                'Low Target Price': low,
                'Recommendation Mean': recommendation_mean,
                'Avg Price Target Snapshot': avg_price_snapshot,
                'Avg Recommendation Snapshot': avg_recommendation_snapshot,
                'Recommendation Trend': recommendation_trend
            })

        # Creating a DataFrame for comparison
        comparison_df = pd.DataFrame(comparison_results)

        # Finding the best stock based on a weighted average of price and recommendation (Mean and Trend)
        comparison_df['Investment Score'] = comparison_df[
            ['Mean Target Price', 'Recommendation Mean', 'Recommendation Trend']].mean(axis=1)

        # Sorting the results by the highest investment score
        best_stock = comparison_df.sort_values(by='Investment Score', ascending=False).iloc[0]

        return comparison_df, best_stock


    # Streamlit interface for Stock Comparison
    st.title("Stock Comparison Tool")
    st.write("Select the stocks you want to compare:")

    # Ensure user selects stocks from 'selected_stocks' multiselect
    if selected_stocks:
        stocks_data = []

        for stock in selected_stocks:
            # Assuming get_stock_comparison_data fetches the comparison data for each stock
            stock_data = get_stock_comparison_data(stock)
            if stock_data:
                stocks_data.append({**stock_data, 'stock_id': stock})

        if stocks_data:
            # Perform comparison if there are any stocks selected
            comparison_df, best_stock = compare_stocks(stocks_data)

            # Display the comparison results
            st.subheader("Stock Comparison Results")
            st.dataframe(comparison_df)

            # Display the best stock
            st.subheader("Best Stock to Invest in Currently:")
            st.write(
                f"The best stock to invest in currently is **{best_stock['Stock ID']}** based on the comparison metrics.")

            # Display detailed information about the best stock
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