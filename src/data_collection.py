import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Fetch historical data
data = yf.download("AAPL", start="2010-01-01", end="2025-01-01")
data = data[['Close']]  # Extract only the 'Close' prices
data.dropna(inplace=True)  # Drop missing values
print(data.head())

# Scale data to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

