from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),  # 60 time steps, 1 feature
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for price prediction
])
model.compile(optimizer='adam', loss='mean_squared_error')
