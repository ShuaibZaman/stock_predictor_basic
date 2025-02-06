import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker.
    :param ticker: Stock ticker (e.g., 'AAPL').
    :param start_date: Start date for data (YYYY-MM-DD).
    :param end_date: End date for data (YYYY-MM-DD).
    :return: DataFrame with stock prices.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']].dropna()  # Extract 'Close' prices and drop missing values
    return data


