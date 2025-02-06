import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data, sequence_length=60):
    """
    Normalize data, create sequences, and split into train/test sets.
    :param data: DataFrame with stock prices.
    :param sequence_length: Number of time steps in each sequence.
    :return: Train and test sets (X_train, X_test, y_train, y_test).
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])  # Past sequence_length days
        y.append(scaled_data[i, 0])  # Target is the next day's price
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
