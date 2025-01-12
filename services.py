import requests

# Define headers for the API calls
headers = {'X-Api-Key': 'sk-live-nudh24voq1sg4F98bFJXvy4Hh5PjnwZEDpHq11xb'}


# Fetch most active stocks data from the API (NSE or BSE)
def get_most_active_stocks(exchange='BSE'):
    url = f'https://stock.indianapi.in/{exchange}_most_active'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for successful response
        return response.json()  # Return JSON response
    except requests.exceptions.RequestException:
        return None


# Fetch upcoming IPOs data from the API
def get_upcoming_ipos():
    url = 'https://stock.indianapi.in/ipo'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get('upcoming', [])
    except requests.exceptions.RequestException:
        return []


# Fetch recent announcements for a selected stock
def get_recent_announcements(stock_name):
    url = f'https://stock.indianapi.in/recent_announcements?stock_name={stock_name}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []


# Fetch historical stats for a selected stock
def get_historical_stats(stock_name):
    url = f'https://stock.indianapi.in/historical_stats?stock_name={stock_name}&stats=all'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# Fetch stock information from the API
def fetch_stock_info(stock_name):
    url = f'https://stock.indianapi.in/stock?name={stock_name}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


# Function to get stock data from the API
def get_stock_comparison_data(stock_id):
    url = f'https://stock.indianapi.in/stock_target_price?stock_id={stock_id}'
    headers = {'X-Api-Key': 'sk-live-nudh24voq1sg4F98bFJXvy4Hh5PjnwZEDpHq11xb'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None
