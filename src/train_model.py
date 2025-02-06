from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape):
    """
    Build the LSTM model.
    :param input_shape: Shape of input data (time steps, features).
    :return: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the LSTM model.
    :param model: Compiled LSTM model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param epochs: Number of epochs.
    :param batch_size: Batch size.
    :return: Trained model.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
