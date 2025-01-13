## StockiFyi(Stocks Forecasting Application)

### Overview

The Stock Forecasting Application is a powerful tool designed to help users analyze and predict stock price trends using advanced forecasting models. It integrates machine learning models like Prophet and ARIMA to provide users with insightful forecasts based on historical stock data. The application is built with a user-friendly interface powered by Streamlit, making it accessible for both novice and expert users in the financial domain.

###  Features

- Multiple Forecasting Models: Users can select between Prophet and ARIMA models for forecasting, each offering unique benefits in time series prediction.
- Customizable Forecasting Period: Allows users to set a specific number of days for forecasting, providing flexibility in short-term or long-term predictions.
- Visualizations: Generates clear and interactive plots of the forecasted stock prices, including confidence intervals, to help users understand potential price movements.
- Performance Metrics: Provides key metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for model performance evaluation.
- Historical Data Analysis: Displays past stock performance, helping users correlate historical trends with forecasted data.
- Comparison Tool: Enables users to compare multiple stocks based on various financial metrics, helping them make informed investment decisions.
- Upcoming IPOs: A sidebar feature that lists upcoming Initial Public Offerings (IPOs) with relevant details like bid start dates and listing dates.

###  How It Works

1. Data Input: The application fetches historical stock data using the `create_dummy_data` function, which simulates stock prices for a given period.
2. Model Selection: Users can choose between Prophet and ARIMA for forecasting.
   - Prophet: Developed by Facebook, suitable for handling seasonality and trend changes in the data.
   - ARIMA: A traditional time series model ideal for datasets with trends and no clear seasonality.
3. Forecasting: Based on the selected model and input parameters (start date, end date, forecast period), the app forecasts future stock prices.
4. Visualization and Metrics: The results are plotted and key forecasting metrics are displayed to give users a comprehensive view of the model's accuracy and reliability.
5. Stock Comparison: Users can compare different stocks based on various financial metrics to identify the best investment opportunities.

###  Technology Stack

- Frontend: Streamlit for interactive UI components and real-time data visualization.
- Backend: Python, using libraries such as:
  - Prophet: For advanced time series forecasting.
  - statsmodels: For ARIMA model implementation.
  - matplotlib & seaborn: For data visualization.
  - pandas & numpy: For data manipulation and numerical operations.
  - sklearn: For evaluating model performance with metrics like MAE, MSE, and RMSE.


###  Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-forecasting-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-forecasting-app
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

###  Usage

1. Launch the application in your web browser.
2. Select the "Forecasting" tab to start forecasting stock prices.
3. Choose a stock, set the desired forecast period, and select the forecasting model (Prophet or ARIMA).
4. View the forecasted results, performance metrics, and visualizations.
5. Navigate to the "Comparison" tab to compare various stocks based on financial metrics.
6. Check the sidebar for information on upcoming IPOs.

---

###  Setup Instructions for Running in Cloud AWS EC2

Follow these steps to set up the environment and run the application:

### Update and Upgrade System Packages
```bash
sudo yum update -y
sudo yum upgrade -y
sudo yum makecache
```

### Install Essential Packages
```bash
sudo yum install git curl unzip tar make sudo vim wget -y
```

### Clone the Repository
```bash
git clone "https://github.com/nitesh318/StockiFyi.git"
```

### Install Python and pip
```bash
sudo yum install python3 -y
sudo yum install python3-pip -y
```

### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### Install Additional Python Packages
```bash
pip install streamlit==1.41.1
pip install streamlit-option-menu
pip install matplotlib==3.10.0
pip install pandas==2.2.3
pip install numpy==2.2.1
pip install scikit-learn==1.6.1
pip install statsmodels==0.14.4
pip install plotly==5.24.1
pip install requests==2.32.3
pip install prophet==1.1.6
```

### Run the Streamlit App
```bash
python3 -m streamlit run app.py
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
